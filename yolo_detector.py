# -*- coding: utf-8 -*-
import numpy as np
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime
import pandas as pd
from collections import defaultdict, deque
import os
import math
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import threading
import subprocess
import sys
import webbrowser
from PIL import Image, ImageDraw, ImageFont

# 상수 정의 (매직 넘버 제거)
class Constants:
    # 거리 및 속도 관련
    DISTANCE_THRESHOLD_PIXELS = 10  # 점과 선 사이의 거리 임계값
    DRAG_DETECTION_RADIUS = 15     # 드래그 감지 반경
    ACCEL_LIMIT_MPS2 = 50          # 비현실적인 가속도 제한값
    
    # 성능 관련
    DISTANCE_CALC_INTERVAL = 5      # N프레임마다 거리 계산
    SPEED_CACHE_SIZE = 100          # 속도 데이터 캐시 크기
    
    # YOLO 및 추적 관련
    DEFAULT_IMGSZ = 1280           # 기본 이미지 크기
    CONFIDENCE_THRESHOLD = 0.2     # YOLO 신뢰도 임계값
    FILTER_THRESHOLD = 0.15        # 검지 필터링 임계값
    IOU_THRESHOLD = 0.4            # IOU 임계값
    
    # 추적기 설정
    TRACKER_MAX_AGE = 10
    TRACKER_MIN_HITS = 3
    TRACKER_IOU_THRESHOLD = 0.3
    
    # 가려짐 검지 관련 상수
    EDGE_MARGIN_RATIO = 0.03      # 프레임 가장자리 마진 비율
    EDGE_MARGIN_PX = 16           # 프레임 가장자리 마진 픽셀
    CORE_SHRINK_RATIO = 0.06      # ROI 테두리 제거 비율
    EDGE_CUT_DELTA = 0.20         # 엣지컷 판단 임계값
    EMA_ALPHA = 0.5               # EMA 평활화 계수
    ALARM_OCC_THRESHOLD = 0.50    # 경보 가려짐 임계값
    ALARM_MIN_FRAMES = 3          # 연속 경보 최소 프레임

# 전역 변수 초기화
estimated_vertical_m = None
estimated_horizontal_m = None
lane_polygons = []
lane_thresholds = {}
custom_regions = []
current_custom_region = []
custom_region_mode = False
tracking_lines = []
dragging_point = None
dragging_index = None

# 항공뷰 모드 관련 전역 변수
aerial_view_mode = False
aerial_meters_per_pixel = None

# 주행모드 관련 전역 변수
driving_mode = False
detected_signs = {}  # 검지된 표지판 저장
sign_screenshots_saved = {}  # 스크린샷 저장 여부 추적

# 가려짐 검지 관련 전역 변수
occlusion_tracker = {}  # 트랙 ID별 가려짐 EMA 및 경보 상태
occlusion_events = []   # 가려짐 이벤트 로그

# Tail 추적 기능 관련 전역 변수
show_trails = False  # 't' 키로 토글
show_accumulated_trails = False  # 'i' 키로 토글 (모든 객체의 누적 경로)
accumulated_trail_paths = {}  # 모든 객체의 전체 이동 경로 저장

# === 가려짐 검지 관련 함수들 ===

def compute_occ_ratio_roi(roi_bgr):
    """
    ROI에서 가려짐 비율을 계산
    Args:
        roi_bgr: BGR 이미지 ROI
    Returns:
        (occ_ratio, method): 가려짐 비율(0~1)과 분석 방법
    """
    try:
        h, w = roi_bgr.shape[:2]
        if h < 20 or w < 20:
            return 0.5, "fallback"  # 너무 작은 ROI
        
        # Grayscale 변환
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # 너무 어두운 이미지 체크
        mean_brightness = np.mean(gray)
        if mean_brightness < 30:
            return 0.5, "fallback"  # 너무 어두움
        
        # Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny Edge Detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.5, "fallback"  # 컨투어 없음
        
        # 가장 큰 컨투어 선택
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 컨투어 면적이 너무 작으면 fallback
        contour_area = cv2.contourArea(largest_contour)
        if contour_area < (w * h * 0.1):  # ROI 면적의 10% 미만
            return 0.5, "fallback"
        
        # 원형 vs 다각형 판단 (둘레 길이 기준)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * contour_area / (perimeter ** 2) if perimeter > 0 else 0
        
        if circularity > 0.6:  # 원형에 가까움
            return _compute_circle_visibility(edges, largest_contour, w, h)
        else:  # 다각형
            return _compute_polygon_visibility(edges, largest_contour, w, h)
            
    except Exception as e:
        print(f"가려짐 계산 오류: {e}")
        return 0.5, "fallback"

def _compute_circle_visibility(edges, contour, w, h):
    """원형 객체의 가시성 계산"""
    try:
        # 최소외접원
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < 10:
            return 0.5, "fallback"
        
        # 원둘레에서 샘플링
        num_samples = max(int(2 * np.pi * radius / 3), 20)  # 3픽셀마다 하나씩
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        
        visible_count = 0
        for angle in angles:
            x = int(cx + radius * np.cos(angle))
            y = int(cy + radius * np.sin(angle))
            
            # 범위 체크
            if 0 <= x < w and 0 <= y < h:
                # 주변 3x3 영역에서 엣지 확인
                edge_found = False
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h and edges[ny, nx] > 0:
                            edge_found = True
                            break
                    if edge_found:
                        break
                
                if edge_found:
                    visible_count += 1
        
        visible_ratio = visible_count / len(angles) if len(angles) > 0 else 0
        occ_ratio = 1 - visible_ratio
        return occ_ratio, "circle"
        
    except Exception:
        return 0.5, "fallback"

def _compute_polygon_visibility(edges, contour, w, h):
    """다각형 객체의 가시성 계산"""
    try:
        # 기대 경계 마스크 생성
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(mask, [contour], True, 255, 2)
        
        # 경계와 엣지의 교집합
        boundary_pixels = np.where(mask > 0)
        total_boundary = len(boundary_pixels[0])
        
        if total_boundary == 0:
            return 0.5, "fallback"
        
        # 경계 위치에서 엣지 확인
        visible_count = 0
        for i in range(total_boundary):
            y, x = boundary_pixels[0][i], boundary_pixels[1][i]
            if edges[y, x] > 0:
                visible_count += 1
        
        visible_ratio = visible_count / total_boundary
        occ_ratio = 1 - visible_ratio
        return occ_ratio, "poly"
        
    except Exception:
        return 0.5, "fallback"

def is_near_edge(box, W, H, margin_ratio=None, margin_px=None):
    """
    박스가 프레임 가장자리 근처에 있는지 판단
    Args:
        box: (x1, y1, x2, y2)
        W, H: 프레임 크기
        margin_ratio: 가장자리 마진 비율
        margin_px: 가장자리 마진 픽셀
    Returns:
        (near_edge, distance): 가장자리 근접 여부와 최소 거리
    """
    if margin_ratio is None:
        margin_ratio = Constants.EDGE_MARGIN_RATIO
    if margin_px is None:
        margin_px = Constants.EDGE_MARGIN_PX
    
    x1, y1, x2, y2 = box
    
    # 마진 계산 (비율과 픽셀 중 큰 값)
    margin_w = max(W * margin_ratio, margin_px)
    margin_h = max(H * margin_ratio, margin_px)
    
    # 각 가장자리까지의 거리
    dist_left = x1
    dist_right = W - x2
    dist_top = y1
    dist_bottom = H - y2
    
    min_distance = min(dist_left, dist_right, dist_top, dist_bottom)
    
    # 가장자리 근처 판단
    near_edge = (dist_left < margin_w or dist_right < margin_w or 
                dist_top < margin_h or dist_bottom < margin_h)
    
    return near_edge, min_distance

def compute_occ_core(roi_bgr, shrink_ratio=None):
    """
    ROI 테두리를 제거한 후 가려짐 비율 계산
    Args:
        roi_bgr: BGR 이미지 ROI
        shrink_ratio: 테두리 제거 비율
    Returns:
        occ_core: 코어 영역의 가려짐 비율 (None if 실패)
    """
    if shrink_ratio is None:
        shrink_ratio = Constants.CORE_SHRINK_RATIO
    
    try:
        h, w = roi_bgr.shape[:2]
        
        # 축소할 픽셀 수 계산
        shrink_w = int(w * shrink_ratio)
        shrink_h = int(h * shrink_ratio)
        
        # 축소된 영역이 너무 작으면 None 반환
        if w - 2*shrink_w < 10 or h - 2*shrink_h < 10:
            return None
        
        # 코어 영역 추출
        core_roi = roi_bgr[shrink_h:h-shrink_h, shrink_w:w-shrink_w]
        
        # 코어 영역의 가려짐 비율 계산
        occ_core, _ = compute_occ_ratio_roi(core_roi)
        return occ_core
        
    except Exception:
        return None

def severity_from_occ(occ):
    """가려짐 정도에 따른 등급 반환"""
    if occ < 0.15:
        return "정상"
    elif occ < 0.40:
        return "부분"
    else:
        return "심함"

def ema_update(prev_value, new_value, alpha=None):
    """EMA (Exponential Moving Average) 업데이트"""
    if alpha is None:
        alpha = Constants.EMA_ALPHA
    return alpha * new_value + (1 - alpha) * prev_value

def get_severity_color(severity):
    """등급별 색상 반환 (BGR)"""
    colors = {
        "정상": (0, 255, 0),      # 초록
        "부분": (0, 255, 255),    # 노랑
        "심함": (0, 0, 255)       # 빨강
    }
    return colors.get(severity, (255, 255, 255))  # 기본값: 흰색

def run_image_test_mode(test_images, model_path, output_folder_path, current_datetime):
    """이미지 기반 테스트 모드 실행"""
    try:
        print(f"🧪 이미지 테스트 모드 시작")
        print(f"   모델: {model_path}")
        print(f"   이미지 수: {len(test_images)}")
        
        # YOLO 모델 로드
        yolo_model = YOLO(model_path)
        class_names = yolo_model.names if hasattr(yolo_model, 'names') else {}
        
        test_results = []
        
        # 모드별 테스트 분할
        modes_to_test = ["차량 검지 (항공뷰/호모그래피용)", "표지판 검지 (주행모드용)", "가려짐 분석"]
        
        for img_idx, image_path in enumerate(test_images):
            print(f"\n📷 이미지 {img_idx + 1}/{len(test_images)}: {os.path.basename(image_path)}")
            
            # 이미지 로드
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"❌ 이미지 로드 실패: {image_path}")
                continue
            
            H, W = frame.shape[:2]
            
            # YOLO 검지
            try:
                results = yolo_model.predict(source=frame, 
                                           imgsz=Constants.DEFAULT_IMGSZ, 
                                           conf=Constants.CONFIDENCE_THRESHOLD, 
                                           iou=Constants.IOU_THRESHOLD)
                
                detections = []
                for result_item in results:
                    boxes = result_item.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_idx = int(box.cls[0])
                        class_name = class_names.get(cls_idx, f"Unknown_{cls_idx}")
                        detections.append([x1, y1, x2, y2, conf, cls_idx, class_name])
                
                print(f"   검지된 객체: {len(detections)}개")
                
                # 결과 이미지 생성
                result_frame = frame.copy()
                
                # 모드별 분석
                for detection in detections:
                    x1, y1, x2, y2, conf, cls_idx, class_name = detection
                    
                    # 1. 차량 검지 테스트 (항공뷰/호모그래피 모드용)
                    vehicle_classes = ['car', 'truck', 'bus', 'motorbike', 'bicycle', 'van']
                    is_vehicle = class_name.lower() in vehicle_classes
                    
                    # 2. 표지판 검지 테스트 (주행모드용)
                    sign_classes = ['traffic_sign', 'stop_sign', 'yield_sign', 'speed_limit', 'no_entry', 'traffic_light', 'warning_sign']
                    is_sign = any(sign_class in class_name.lower() for sign_class in sign_classes)
                    
                    # 3. 가려짐 분석 (표지판에 대해서만)
                    occ_ratio, occ_method = 0.0, "none"
                    severity = "정상"
                    if is_sign:
                        try:
                            roi = frame[int(y1):int(y2), int(x1):int(x2)]
                            if roi.size > 0:
                                occ_ratio, occ_method = compute_occ_ratio_roi(roi)
                                severity = severity_from_occ(occ_ratio)
                        except Exception as e:
                            print(f"   가려짐 분석 오류: {e}")
                    
                    # 색상 결정
                    if is_sign:
                        color = get_severity_color(severity)  # 가려짐 정도별 색상
                    elif is_vehicle:
                        color = (255, 0, 0)  # 파랑 (차량)
                    else:
                        color = (128, 128, 128)  # 회색 (기타)
                    
                    # 바운딩 박스 그리기
                    cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # 라벨 생성 (한글)
                    labels = [f"{class_name}: {conf:.2f}"]
                    
                    if is_vehicle:
                        labels.append("차량")
                    elif is_sign:
                        labels.append(f"표지판 | 가림:{occ_ratio:.2f} ({severity}) [{occ_method}]")
                    else:
                        labels.append("기타")
                    
                    # 라벨 표시 (PIL 사용으로 한글 지원)
                    y_offset = int(y1) - 10
                    for i, label in enumerate(labels):
                        label_y = y_offset - i * 25
                        if label_y < 25:
                            label_y = int(y2) + 25 + i * 25
                        
                        result_frame = put_text_pil(result_frame, label, (int(x1), label_y), 
                                                   font_size=16, color=color)
                    
                    # 테스트 결과 기록
                    test_results.append({
                        'image_name': os.path.basename(image_path),
                        'class_name': class_name,
                        'confidence': conf,
                        'is_vehicle': is_vehicle,
                        'is_sign': is_sign,
                        'occ_ratio': occ_ratio,
                        'occ_severity': severity,
                        'occ_method': occ_method,
                        'bbox': (x1, y1, x2, y2)
                    })
                
                # 결과 이미지 저장
                result_filename = f"test_result_{img_idx+1:03d}_{os.path.basename(image_path)}"
                result_path = os.path.join(output_folder_path, result_filename)
                cv2.imwrite(result_path, result_frame)
                
                print(f"   ✅ 결과 저장: {result_filename}")
                
                # 실시간 이미지 표시
                display_frame = result_frame.copy()
                
                # 화면 크기에 맞게 리사이즈
                screen_height = 800  # 최대 높이
                if display_frame.shape[0] > screen_height:
                    scale = screen_height / display_frame.shape[0]
                    new_width = int(display_frame.shape[1] * scale)
                    display_frame = cv2.resize(display_frame, (new_width, screen_height))
                
                # 정보 오버레이 추가
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                
                # 텍스트 정보 표시 (PIL 사용으로 한글 지원)
                info_texts = [
                    f"이미지 {img_idx + 1}/{len(test_images)}: {os.path.basename(image_path)}",
                    f"전체 검지: {len(detections)}개",
                    f"차량: {sum(1 for d in detections if d[6].lower() in ['car', 'truck', 'bus', 'motorbike', 'bicycle', 'van'])}개",
f"표지판: {sum(1 for d in detections if any(s in d[6].lower() for s in ['traffic_sign', 'stop_sign', 'yield_sign', 'speed_limit', 'no_entry', 'traffic_light', 'warning_sign']))}개"
                ]
                
                # PIL을 사용해서 한글 텍스트 그리기
                for i, text in enumerate(info_texts):
                    display_frame = put_text_pil(display_frame, text, (15, 30 + i * 25), font_size=20, color=(255, 255, 255))
                
                # OpenCV 창에 표시
                cv2.namedWindow('Test Results', cv2.WINDOW_NORMAL)
                cv2.imshow('Test Results', display_frame)
                
                # 키 입력 대기
                print(f"   👀 결과 확인 중... (아무 키나 누르면 다음 이미지로, 'q'를 누르면 종료)")
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' 또는 ESC 키
                    print(f"   🛑 사용자가 테스트를 중단했습니다.")
                    cv2.destroyAllWindows()
                    break
                
                cv2.destroyWindow('Test Results')
                
            except Exception as e:
                print(f"   ❌ YOLO 검지 실패: {e}")
        
        # 테스트 결과 CSV 저장
        if test_results:
            df_test = pd.DataFrame(test_results)
            
            # 통계 계산
            total_detections = len(df_test)
            vehicle_detections = len(df_test[df_test['is_vehicle'] == True])
            sign_detections = len(df_test[df_test['is_sign'] == True])
            occluded_signs = len(df_test[(df_test['is_sign'] == True) & (df_test['occ_ratio'] > 0.15)])
            
            test_csv_path = os.path.join(output_folder_path, 
                f"{current_datetime}_image_test_results.csv")
            df_test.to_csv(test_csv_path, index=False, encoding='utf-8-sig')
            
            # 테스트 요약 생성
            summary_text = []
            summary_text.append("=== 이미지 테스트 결과 요약 ===")
            summary_text.append(f"테스트 이미지: {len(test_images)}장")
            summary_text.append(f"총 검지 객체: {total_detections}개")
            summary_text.append(f"")
            summary_text.append("== 모드별 검지 성능 ==")
            summary_text.append(f"차량 검지 (항공뷰/호모그래피): {vehicle_detections}개")
            summary_text.append(f"표지판 검지 (주행모드): {sign_detections}개")
            summary_text.append(f"")
            summary_text.append("== 가려짐 분석 결과 ==")
            summary_text.append(f"분석된 표지판: {sign_detections}개")
            summary_text.append(f"가려진 표지판: {occluded_signs}개")
            
            if sign_detections > 0:
                occlusion_rate = (occluded_signs / sign_detections) * 100
                summary_text.append(f"가려짐 비율: {occlusion_rate:.1f}%")
            
            summary_text.append(f"")
            summary_text.append(f"사용된 모델: {os.path.basename(model_path)}")
            
            summary_path = os.path.join(output_folder_path, 
                f"{current_datetime}_image_test_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_text))
            
            print(f"\n✅ 테스트 완료!")
            print(f"   결과 CSV: {test_csv_path}")
            print(f"   요약 파일: {summary_path}")
            
            # 모든 OpenCV 창 닫기
            cv2.destroyAllWindows()
            
            # 결과 폴더 자동으로 열기
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(output_folder_path)
                elif os.name == 'posix':  # macOS/Linux
                    subprocess.run(['open', output_folder_path], check=True)
                print(f"📂 결과 폴더를 열었습니다: {output_folder_path}")
            except Exception as e:
                print(f"⚠️ 폴더 열기 실패: {e}")
            
            messagebox.showinfo("테스트 완료", 
                               f"이미지 테스트가 완료되었습니다!\n\n"
                               f"결과 저장 위치: {output_folder_path}\n"
                               f"• 테스트 이미지: {len(test_images)}장\n"
                               f"• 총 검지 객체: {total_detections}개\n"
                               f"• 차량 검지: {vehicle_detections}개\n"
                               f"• 표지판 검지: {sign_detections}개\n"
                               f"• 가려진 표지판: {occluded_signs}개\n\n"
                               f"📂 결과 폴더가 자동으로 열렸습니다!")
        else:
            cv2.destroyAllWindows()
            
            # 검지된 객체가 없어도 폴더 열기
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(output_folder_path)
                elif os.name == 'posix':  # macOS/Linux
                    subprocess.run(['open', output_folder_path], check=True)
                print(f"📂 결과 폴더를 열었습니다: {output_folder_path}")
            except Exception as e:
                print(f"⚠️ 폴더 열기 실패: {e}")
            
            messagebox.showinfo("테스트 완료", 
                               f"검지된 객체가 없습니다.\n\n"
                               f"결과 저장 위치: {output_folder_path}\n"
                               f"📂 결과 폴더가 자동으로 열렸습니다!")
            
    except Exception as e:
        print(f"❌ 테스트 모드 오류: {e}")
        cv2.destroyAllWindows()  # 오류 시에도 창 닫기
        messagebox.showerror("오류", f"테스트 중 오류가 발생했습니다:\n{e}")

def simple_iou_matching(detections, prev_boxes, iou_threshold=0.5):
    """
    간단한 IoU 기반 매칭 (트래킹이 없을 때 사용)
    Args:
        detections: [(x1,y1,x2,y2,conf,cls_id), ...]
        prev_boxes: {track_id: (x1,y1,x2,y2), ...}
        iou_threshold: IoU 임계값
    Returns:
        matches: {detection_idx: track_id, ...}
    """
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    matches = {}
    used_track_ids = set()
    
    for det_idx, detection in enumerate(detections):
        det_box = detection[:4]  # (x1,y1,x2,y2)
        best_iou = 0
        best_track_id = None
        
        for track_id, prev_box in prev_boxes.items():
            if track_id in used_track_ids:
                continue
            
            iou = calculate_iou(det_box, prev_box)
            if iou > best_iou and iou > iou_threshold:
                best_iou = iou
                best_track_id = track_id
        
        if best_track_id is not None:
            matches[det_idx] = best_track_id
            used_track_ids.add(best_track_id)
    
    return matches

def put_text_pil(frame, text, position, font_size=20, color=(255,255,255)):
    """다양한 OS에서 호환되는 폰트 사용"""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # OS별 폰트 경로 설정
    font_paths = []
    if sys.platform == "win32":
        font_paths = [
            "C:/Windows/Fonts/malgun.ttf",      # 한글
            "C:/Windows/Fonts/arial.ttf",       # 영어
            "C:/Windows/Fonts/NanumGothic.ttf", # 나눔고딕
        ]
    elif sys.platform == "darwin":  # macOS
        font_paths = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/System/Library/Fonts/Arial.ttf",
        ]
    else:  # Linux
        font_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    
    # 사용 가능한 폰트 찾기
    font = None
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        except (IOError, OSError):
            continue
    
    # 모든 폰트 실패 시 기본 폰트 사용
    if font is None:
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_multiline_text(frame, text, position, font_size=20, color=(255,255,255)):
    lines = text.split('\n')
    y = position[1]
    for line in lines:
        frame = put_text_pil(frame, line, (position[0], y), font_size, color)
        y += font_size + 5
    return frame

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def calculate_aerial_distance(point1, point2):
    """항공뷰 모드에서 두 점 사이의 실제 거리 계산 (미터)"""
    global aerial_meters_per_pixel
    if aerial_meters_per_pixel is None:
        return None
    
    pixel_distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    return pixel_distance * aerial_meters_per_pixel

def sanitize_filename(filename, max_length=100):
    """Windows 호환 파일명 생성 (최대 150자 제한)"""
    # Windows 금지 문자 제거: <>:"/\|?*[]
    forbidden_chars = '<>:"/\\|?*[]'
    for char in forbidden_chars:
        filename = filename.replace(char, '')
    
    # 특수 문자를 언더스코어로 변경
    import re
    filename = re.sub(r'[^\w\s.-]', '_', filename)
    
    # 연속된 공백을 언더스코어로 변경
    filename = re.sub(r'\s+', '_', filename)
    
    # 연속된 언더스코어를 하나로 정리
    while '__' in filename:
        filename = filename.replace('__', '_')
    
    # 앞뒤 공백과 언더스코어 제거
    filename = filename.strip(' _')
    
    # 길이 제한 (확장자 고려)
    if len(filename) > max_length:
        # 파일명에서 확장자 분리
        if '.' in filename:
            name_part, ext_part = filename.rsplit('.', 1)
            available_length = max_length - len(ext_part) - 1  # .확장자 길이 제외
            if available_length > 0:
                filename = name_part[:available_length] + '.' + ext_part
            else:
                filename = filename[:max_length]
        else:
            filename = filename[:max_length]
    
    # 빈 문자열이나 점으로만 구성된 경우 처리
    if not filename or filename.replace('.', '').strip() == '':
        filename = 'output'
    
    return filename

def create_test_output_folder(model_path):
    """테스트 모드용 출력 폴더 생성"""
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = os.path.basename(model_path)
    
    # 파일명 정제
    safe_model_name = sanitize_filename(model_name, 50)
    
    output_folder_name = f"TEST_{safe_model_name}_imonkfcwifi_{current_datetime}"
    output_folder_name = sanitize_filename(output_folder_name, 200)
    
    print(f"테스트 모드 폴더명: {output_folder_name}")
    
    # 현재 디렉토리에 폴더 생성
    current_dir = os.getcwd()
    output_folder_path = os.path.join(current_dir, output_folder_name)
    output_folder_path = os.path.normpath(output_folder_path)
    
    print(f"생성하려는 테스트 폴더 경로: {output_folder_path}")
    
    # 폴더 생성 시도
    try:
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path, exist_ok=True)
            print(f"✅ 테스트 폴더 생성 성공: {output_folder_path}")
        else:
            print(f"⚠️ 폴더가 이미 존재합니다: {output_folder_path}")
    except Exception as e:
        print(f"❌ 테스트 폴더 생성 실패: {e}")
        # 데스크톱 폴더로 대체
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        if os.path.exists(desktop_path):
            output_folder_path = os.path.join(desktop_path, output_folder_name)
            try:
                os.makedirs(output_folder_path, exist_ok=True)
                print(f"✅ 데스크톱에 테스트 폴더 생성: {output_folder_path}")
            except:
                print(f"❌ 데스크톱에도 폴더 생성 실패. 현재 디렉토리 사용.")
                output_folder_path = current_dir
        else:
            output_folder_path = current_dir
    
    return output_folder_path, current_datetime

def create_output_folder(input_video_path, model_path):
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_name = os.path.basename(input_video_path)
    model_name = os.path.basename(model_path)
    video_name_without_ext = os.path.splitext(video_name)[0]
    
    # 파일명 정제
    safe_video_name = sanitize_filename(video_name_without_ext, 100)
    safe_model_name = sanitize_filename(model_name, 50)
    
    output_folder_name = f"{safe_video_name}_{safe_model_name}_imonkfcwifi_{current_datetime}"
    output_folder_name = sanitize_filename(output_folder_name, 200)
    
    print(f"원본 비디오명: {video_name_without_ext}")
    print(f"정제된 비디오명: {safe_video_name}")
    print(f"최종 폴더명: {output_folder_name}")
    
    # Windows 경로 처리 개선
    video_dir = os.path.dirname(os.path.abspath(input_video_path))
    if not video_dir:  # 빈 문자열인 경우
        video_dir = os.getcwd()
    
    output_folder_path = os.path.join(video_dir, output_folder_name)
    # 경로 정규화
    output_folder_path = os.path.normpath(output_folder_path)
    
    print(f"생성하려는 폴더 경로: {output_folder_path}")
    
    # 폴더 생성 시도
    try:
        os.makedirs(output_folder_path, exist_ok=True)
        print(f"폴더 생성 성공: {output_folder_path}")
    except Exception as e:
        print(f"폴더 생성 오류: {e}")
        # 폴더 생성 실패 시 더 간단한 이름으로 fallback
        fallback_name = f"output_{current_datetime}"
        fallback_path = os.path.join(os.getcwd(), fallback_name)
        try:
            os.makedirs(fallback_path, exist_ok=True)
            output_folder_path = fallback_path
            print(f"Fallback 폴더 생성 성공: {output_folder_path}")
        except Exception as e2:
            print(f"Fallback 폴더 생성도 실패: {e2}")
            # 최후의 수단으로 현재 디렉토리 사용
            output_folder_path = os.getcwd()
            print(f"현재 디렉토리 사용: {output_folder_path}")
    
    return output_folder_path, current_datetime, video_name, model_name, video_name_without_ext

def open_folder(folder_path):
    if sys.platform == "win32":
        os.startfile(folder_path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", folder_path])
    else:
        subprocess.Popen(["xdg-open", folder_path])

def yolo_mouse_callback(event, x, y, flags, param):
    global custom_region_mode, current_custom_region, dragging_point, dragging_index, tracking_lines
    global aerial_view_mode  # 항공뷰 모드 확인용
    
    # OpenCV 이벤트 상수값 확인
    cv2_constants = {
        'EVENT_LBUTTONDOWN': cv2.EVENT_LBUTTONDOWN,
        'EVENT_RBUTTONDOWN': cv2.EVENT_RBUTTONDOWN,
        'EVENT_MBUTTONDOWN': cv2.EVENT_MBUTTONDOWN,
        'EVENT_LBUTTONUP': cv2.EVENT_LBUTTONUP,
        'EVENT_RBUTTONUP': cv2.EVENT_RBUTTONUP,
        'EVENT_MOUSEMOVE': cv2.EVENT_MOUSEMOVE
    }
    
    # 상세한 마우스 이벤트 디버깅
    event_names = {
        cv2.EVENT_RBUTTONDOWN: "우클릭",
        cv2.EVENT_MBUTTONDOWN: "휠클릭", 
        cv2.EVENT_LBUTTONDOWN: "좌클릭",
        cv2.EVENT_LBUTTONUP: "좌클릭업",
        cv2.EVENT_RBUTTONUP: "우클릭업",
        cv2.EVENT_MOUSEMOVE: "마우스이동"
    }
    
    # 모든 클릭 이벤트에 대한 상세 로깅
    if event in [cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN, cv2.EVENT_LBUTTONDOWN]:
        event_name = event_names.get(event, f'알 수 없는 이벤트{event}')
        print(f"[마우스 이벤트] {event_name} at ({x}, {y})")
        print(f"[디버그] 이벤트 값: {event}")
        print(f"[OpenCV 상수값] 좌클릭:{cv2.EVENT_LBUTTONDOWN}, 우클릭:{cv2.EVENT_RBUTTONDOWN}, 휠:{cv2.EVENT_MBUTTONDOWN}")
        print(f"[디버그] 모드: {'항공뷰' if aerial_view_mode else '호모그래피'}, 커스텀 영역 모드: {custom_region_mode}")
        print(f"[디버그] 플래그: {flags}")
        
        # 이벤트 값 정확성 검증
        if event == cv2.EVENT_LBUTTONDOWN:
            print("[확인] 정상적인 좌클릭 이벤트")
        elif event == cv2.EVENT_RBUTTONDOWN:
            print("[확인] 정상적인 우클릭 이벤트")
        elif event == cv2.EVENT_MBUTTONDOWN:
            print("[확인] 정상적인 휠클릭 이벤트")
        else:
            print(f"[경고] 예상치 못한 이벤트 값: {event}")
    
    # 커스텀 영역 모드 처리
    if custom_region_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            current_custom_region.append((x, y))
            print(f"[커스텀 영역] 점 추가: ({x}, {y}), 총 {len(current_custom_region)}개 점")
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(current_custom_region) >= 3:
                custom_regions.append(np.array(current_custom_region, dtype=np.int32))
                print(f"[커스텀 영역] 영역 완성: {len(current_custom_region)}개 점으로 구성")
            current_custom_region.clear()
            custom_region_mode = False
            print("[커스텀 영역] 모드 종료")
    else:
        # 일반 모드에서 마우스 이벤트 처리 - 우클릭 또는 휠클릭으로 추적선 생성
        if event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_MBUTTONDOWN:
            click_type = "우클릭" if event == cv2.EVENT_RBUTTONDOWN else "휠클릭"
            print(f"[{click_type} 처리] 좌표: ({x}, {y}), 항공뷰 모드: {aerial_view_mode}")
            
            # 새 추적선 시작 또는 기존 추적선 완성
            if not tracking_lines or len(tracking_lines[-1]["points"]) == 2:
                line_name = f"Tracking Line {len(tracking_lines) + 1}"
                new_line = {
                    "name": line_name, 
                    "points": [(x, y)], 
                    "count_cross": 0, 
                    "crossed_ids": set(), 
                    "type": "tracking", 
                    "persistent_crossed_ids": set()
                }
                tracking_lines.append(new_line)
                print(f"[추적선 생성] {click_type}으로 새 추적선 시작: {line_name} at ({x}, {y})")
            else:
                tracking_lines[-1]["points"].append((x, y))
                line_info = tracking_lines[-1]
                print(f"[추적선 완성] {click_type}으로 {line_info['name']} - 시작{line_info['points'][0]} 끝({x}, {y})")
                
        elif event == cv2.EVENT_LBUTTONDOWN:
            print(f"[좌클릭 처리] 드래그 포인트 검색 at ({x}, {y})")
            
            # 기존 추적선의 점들 중에서 드래그 가능한 점 찾기
            for i, line in enumerate(tracking_lines):
                for j, point in enumerate(line["points"]):
                    distance = dist(point, (x, y))
                    if distance < Constants.DISTANCE_THRESHOLD_PIXELS:
                        dragging_point = point
                        dragging_index = (i, j)
                        print(f"[드래그 시작] {line['name']}의 점 {j} at {point} (거리: {distance:.1f})")
                        return
            print("[좌클릭] 드래그 가능한 점을 찾지 못했습니다")
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging_index is not None:
                line_idx, point_idx = dragging_index
                tracking_lines[line_idx]["points"][point_idx] = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if dragging_point is not None:
                print(f"[드래그 완료] 새 위치: ({x}, {y})")
            dragging_point = None
            dragging_index = None
    
    # 추적선 상태 출력 (우클릭/휠클릭 후에만)
    if event in [cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN]:
        print(f"[상태 확인] tracking_lines 개수: {len(tracking_lines)}")
        for i, line in enumerate(tracking_lines):
            points_info = f"{len(line['points'])}개 점"
            if len(line['points']) == 2:
                points_info += f" {line['points'][0]} -> {line['points'][1]}"
            elif len(line['points']) == 1:
                points_info += f" 시작점: {line['points'][0]}"
            print(f"  {i+1}. {line['name']} ({line['type']}) - {points_info}")

def update_smoothing_settings(speed_smoothing, fps):
    """
    모드 변경 시 스무딩 설정을 동적으로 업데이트하는 함수
    """
    global aerial_view_mode
    
    # 현재 모드에 따라 새로운 maxlen 계산
    if aerial_view_mode:
        new_maxlen = int(fps * 0.3) if fps > 0 else 2  # 항공뷰: 더 짧은 스무딩
    else:
        new_maxlen = int(fps * 1) if fps > 0 else 5    # 호모그래피: 기존 스무딩
    
    # 모든 기존 deque의 maxlen을 새로 설정
    for track_id in list(speed_smoothing.keys()):
        old_data = list(speed_smoothing[track_id])
        speed_smoothing[track_id] = deque(old_data[-new_maxlen:], maxlen=new_maxlen)
    
    # 새로운 defaultdict factory 함수도 업데이트
    speed_smoothing.default_factory = lambda: deque(maxlen=new_maxlen)
    
    return new_maxlen

def analysis_process(input_video_path, model_path, output_folder_path, current_datetime, video_name, model_name, video_name_without_ext, help_text, rect_points, width_m, height_m):
    region_records = []
    region_unique_ids = defaultdict(set)
    global custom_region_mode, current_custom_region, custom_regions, tracking_lines
    global estimated_vertical_m, estimated_horizontal_m
    global aerial_view_mode, aerial_meters_per_pixel
    global show_trails, show_accumulated_trails, accumulated_trail_paths
    global driving_mode, detected_signs, sign_screenshots_saved
    global occlusion_tracker, occlusion_events
    display_info = True
    lane_events = []

    yolo_model = YOLO(model_path)
    # YOLO 모델 클래스 이름 가져오기 (딕셔너리 형태라고 가정하고 .get() 사용)
    if hasattr(yolo_model, 'names'): # v8
        class_names = yolo_model.names
    elif hasattr(yolo_model, 'model') and hasattr(yolo_model.model, 'names'): # 이전 버전 호환성
        class_names = yolo_model.model.names
    else:
        print("경고: 모델에서 클래스 이름을 가져올 수 없습니다. 임시 이름을 사용합니다.")
        # 모델 파일에 따라 이 부분은 직접 확인하고 수정해야 할 수 있습니다.
        # 예시: class_names = {0: 'person', 1: 'bicycle', 2: 'car', ...}
        # 우선은 빈 딕셔너리로 두고, Unknown으로 처리되도록 합니다.
        class_names = {}

    # 모드별 관심 클래스 설정
    if driving_mode:
        # 주행모드: 표지판 클래스 (사용자가 파인튜닝한 모델에 따라 달라질 수 있음)
        desired_classes = ['traffic_sign']
        sign_detection_data = []  # 표지판 검지 데이터 저장
    else:
        # 항공뷰용/호모그래피용 클래스 (VisDrone 기준) - 드론 검출 추가
        desired_classes = ['car', 'truck', 'bus', 'motorbike', 'bicycle', 'van', 'people', 'awning-tricycle', 'tricycle', 'drone', 'uav', 'aircraft', 'helicopter']
    
    # 클래스 이름 변경 매핑 (VAN -> CAR-SUV 등)
    class_name_mapping = {
        'van': 'CAR-SUV',
        'car': 'CAR',
        'truck': 'TRUCK',
        'bus': 'BUS',
        'motorbike': 'MOTORBIKE',
        'bicycle': 'BICYCLE',
        'people': 'PERSON'
    }
    
    def map_class_name(original_class):
        """원본 클래스명을 사용자 정의 이름으로 매핑"""
        return class_name_mapping.get(original_class.lower(), original_class.upper())
    
    def save_sign_screenshot(frame, bbox, class_name, confidence, timestamp, frame_number):
        """표지판 검지 시 스크린샷 저장"""
        try:
            x1, y1, x2, y2 = bbox
            # 바운딩 박스 영역 확장 (컨텍스트 포함)
            h, w = frame.shape[:2]
            margin = 50
            x1_exp = max(0, int(x1) - margin)
            y1_exp = max(0, int(y1) - margin)
            x2_exp = min(w, int(x2) + margin)
            y2_exp = min(h, int(y2) + margin)
            
            # 스크린샷 영역 추출
            screenshot = frame[y1_exp:y2_exp, x1_exp:x2_exp]
            
            # 파일명 생성
            timestamp_str = timestamp.strftime("%H%M%S_%f")[:-3]  # 밀리초까지
            filename = f"sign_{class_name}_{timestamp_str}_frame{frame_number:06d}.png"
            filepath = os.path.join(output_folder_path, filename)
            
            # 스크린샷 저장
            cv2.imwrite(filepath, screenshot)
            print(f"📸 표지판 스크린샷 저장: {filename}")
            
            return filepath
        except Exception as e:
            print(f"스크린샷 저장 오류: {e}")
            return None
    
    def process_sign_detection(frame, detections, timestamp, frame_number):
        """표지판 검지 처리 및 데이터 기록 (가려짐 분석 포함)"""
        H, W = frame.shape[:2]
        
        # 이전 프레임 박스 정보 (간단한 IoU 매칭용)
        if not hasattr(process_sign_detection, 'prev_boxes'):
            process_sign_detection.prev_boxes = {}
        
        current_boxes = {}
        next_track_id = max(process_sign_detection.prev_boxes.keys(), default=0) + 1
        
        for det_idx, detection in enumerate(detections):
            x1, y1, x2, y2, conf, cls_id = detection
            
            if cls_id in class_names:
                class_name = class_names[cls_id]
                
                # === 가려짐 분석 시작 ===
                try:
                    # ROI 추출
                    roi = frame[int(y1):int(y2), int(x1):int(x2)]
                    
                    if roi.size > 0:
                        # 1. 가려짐 비율 계산
                        occ_full, method = compute_occ_ratio_roi(roi)
                        
                        # 2. 프레임 가장자리 근처인지 확인
                        near_edge, d_edge = is_near_edge((x1, y1, x2, y2), W, H)
                        
                        # 3. 코어 영역 가려짐 계산
                        occ_core = compute_occ_core(roi)
                        
                        # 4. 엣지컷 판단
                        is_edge_cut = (near_edge and occ_core is not None and 
                                      (occ_full - occ_core >= Constants.EDGE_CUT_DELTA))
                        
                        # 5. 트랙 ID 할당 (간단한 IoU 매칭)
                        matches = simple_iou_matching([detection], process_sign_detection.prev_boxes)
                        
                        if det_idx in matches:
                            track_id = matches[det_idx]
                        else:
                            track_id = next_track_id
                            next_track_id += 1
                        
                        current_boxes[track_id] = (x1, y1, x2, y2)
                        
                        # 6. EMA 업데이트
                        if track_id not in occlusion_tracker:
                            occlusion_tracker[track_id] = {
                                'occ_ema': occ_full,
                                'alarm_count': 0,
                                'last_alarm_frame': -1
                            }
                        else:
                            if not is_edge_cut:  # 엣지컷이 아닐 때만 EMA 업데이트
                                occlusion_tracker[track_id]['occ_ema'] = ema_update(
                                    occlusion_tracker[track_id]['occ_ema'], occ_full)
                        
                        # 7. 경보 처리
                        occ_ema = occlusion_tracker[track_id]['occ_ema']
                        severity = severity_from_occ(occ_ema)
                        
                        # 연속 경보 조건 확인
                        if (not is_edge_cut and occ_ema >= Constants.ALARM_OCC_THRESHOLD):
                            occlusion_tracker[track_id]['alarm_count'] += 1
                            
                            # 연속 N프레임 이상 경보 조건 충족
                            if (occlusion_tracker[track_id]['alarm_count'] >= Constants.ALARM_MIN_FRAMES and
                                frame_number - occlusion_tracker[track_id]['last_alarm_frame'] > 30):  # 1초마다
                                
                                # 가려짐 이벤트 로깅
                                occlusion_event = {
                                    'timestamp': timestamp,
                                    'frame_number': frame_number,
                                    'track_id': track_id,
                                    'class_name': class_name,
                                    'occ_ratio': occ_ema,
                                    'severity': severity,
                                    'method': method,
                                    'is_edge_cut': is_edge_cut,
                                    'bbox': (x1, y1, x2, y2)
                                }
                                occlusion_events.append(occlusion_event)
                                occlusion_tracker[track_id]['last_alarm_frame'] = frame_number
                                
                                print(f"⚠️ 가려짐 경보: {class_name} (Track {track_id}) - occ:{occ_ema:.2f} ({severity})")
                        else:
                            occlusion_tracker[track_id]['alarm_count'] = 0
                        
                        # 8. 오버레이 표시
                        color = get_severity_color(severity)
                        
                        # 바운딩 박스 그리기
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                        
                        # 라벨 생성 (한글)
                        edge_suffix = " [가장자리]" if is_edge_cut else f" [{method}]"
                        occ_label = f"가림:{occ_ema:.2f} ({severity}){edge_suffix}"
                        sign_label = f"{class_name}: {conf:.2f}"
                        
                        # 라벨 표시 (PIL 사용으로 한글 지원)
                        y_offset = int(y1) - 35
                        frame = put_text_pil(frame, sign_label, (int(x1), y_offset), 
                                           font_size=18, color=color)
                        frame = put_text_pil(frame, occ_label, (int(x1), y_offset + 25), 
                                           font_size=16, color=color)
                        
                except Exception as e:
                    print(f"가려짐 분석 오류: {e}")
                    # 오류 시 기본 표시
                    color = (0, 255, 255)  # 노란색
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    label = f"{class_name}: {conf:.2f}"
                    frame = put_text_pil(frame, label, (int(x1), int(y1) - 5), 
                                       font_size=18, color=color)
                # === 가려짐 분석 끝 ===
                
                # 기존 스크린샷 및 데이터 기록 로직
                sign_key = f"{class_name}_{frame_number//30}"  # 30프레임(1초) 단위로 그룹화
                
                if sign_key not in sign_screenshots_saved:
                    # 스크린샷 저장
                    screenshot_path = save_sign_screenshot(frame, [x1, y1, x2, y2], class_name, conf, timestamp, frame_number)
                    
                    # 검지 데이터 기록
                    sign_data = {
                        'timestamp': timestamp,
                        'frame_number': frame_number,
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox_x1': x1,
                        'bbox_y1': y1,
                        'bbox_x2': x2,
                        'bbox_y2': y2,
                        'screenshot_path': screenshot_path
                    }
                    
                    sign_detection_data.append(sign_data)
                    sign_screenshots_saved[sign_key] = True
                    
                    print(f"🚦 새로운 표지판 검지: {class_name} (신뢰도: {conf:.2f})")
        
        # 이전 프레임 박스 정보 업데이트
        process_sign_detection.prev_boxes = current_boxes
    
    print(f"YOLO 모델의 클래스 목록 (또는 감지된 ID): {class_names}")
    print(f"클래스 이름 매핑: VAN -> CAR-SUV, CAR -> CAR 등")


    tracker = DeepSort(max_age=Constants.TRACKER_MAX_AGE, 
                       n_init=Constants.TRACKER_MIN_HITS,
                       max_iou_distance=Constants.TRACKER_IOU_THRESHOLD,
                       max_cosine_distance=0.4,
                       nn_budget=None,
                       embedder="mobilenet")
    # 안전한 파일명 생성을 위해 특수문자 제거
    def clean_filename(filename):
        # Windows에서 허용되지 않는 문자들 제거
        import re
        # 특수문자를 언더스코어로 대체
        filename = re.sub(r'[<>:"/\\|?*\[\]]', '_', filename)
        # 연속된 언더스코어를 하나로 합치기
        filename = re.sub(r'_+', '_', filename)
        # 파일명 길이 제한 (확장자 제외하고 100자)
        if len(filename) > 100:
            filename = filename[:100]
        return filename.strip('_')
    
    clean_video_name = clean_filename(video_name)
    clean_model_name = clean_filename(model_name)
    output_video_path = os.path.join(output_folder_path, f"{current_datetime}_{clean_model_name}_{clean_video_name}_video.avi")
    
    print(f"🎬 비디오 출력 경로: {output_video_path}")
    print(f"   원본 비디오명: {video_name}")
    print(f"   정리된 비디오명: {clean_video_name}")
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        messagebox.showerror("오류", f"영상 파일을 열 수 없습니다: {input_video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0:
        print("경고: 비디오 FPS가 0입니다. 기본값 30으로 설정합니다.")
        fps = 30.0


    # 다양한 코덱 시도 (Windows 호환성 우선순위)
    codecs_to_try = [
        ('MJPG', 'avi'),  # Motion JPEG - 가장 호환성 높음
        ('DIVX', 'avi'),  # DivX 코덱
        ('XVID', 'avi'),  # XviD 코덱
        ('mp4v', 'mp4'),  # MPEG-4
        ('I420', 'avi'),  # Raw YUV 4:2:0
        ('IYUV', 'avi')   # Raw YUV (fallback)
    ]
    
    out = None
    for codec, ext in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            test_path = output_video_path.replace('.avi', f'.{ext}')
            print(f"코덱 {codec} 시도 중... 경로: {test_path}")
            print(f"비디오 설정: {width}x{height} @ {fps} FPS")
            out = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
            if out.isOpened():
                output_video_path = test_path
                print(f"✅ 비디오 출력 설정 성공: {codec} 코덱")
                break
            else:
                print(f"❌ {codec} 코덱 실패")
                out.release()
                out = None
        except Exception as e:
            print(f"❌ {codec} 코덱 오류: {e}")
            if out:
                out.release()
            out = None
            continue
    
    if not out or not out.isOpened():
        print("경고: 비디오 출력을 설정할 수 없습니다. 추가 시도를 진행합니다.")
        print(f"시도한 경로: {output_video_path}")
        print(f"비디오 크기: {width}x{height}, FPS: {fps}")
        
        # 최후의 시도 - 경로 길이 제한으로 파일명 단축
        try:
            video_dir = os.path.dirname(output_video_path)
            # 원본 파일명에서 핵심 부분만 추출 (30자로 제한)
            original_name = os.path.splitext(os.path.basename(input_video_path))[0]
            # 특수문자 제거 및 길이 제한
            safe_name = ''.join(c for c in original_name if c.isalnum() or c in '-_')[:30]
            simple_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{model_name}_{safe_name}_video.mp4"
            simple_path = os.path.join(video_dir, simple_name)
            
            print(f"간단한 경로로 시도: {simple_path}")
            fourcc = cv2.VideoWriter_fourcc('m','p','4','v')  # MP4 호환 코덱
            out = cv2.VideoWriter(simple_path, fourcc, fps, (width, height))
            
            if out.isOpened():
                output_video_path = simple_path
                print(f"✅ 간단한 경로로 성공!")
            else:
                out.release()
                print("❌ 최종 시도 실패 - 비디오 저장 비활성화")
                out = None
                
        except Exception as e:
            print(f"비디오 출력 초기화 오류: {e}")
            out = None

    crossing_data = []
    object_positions_maxlen = max(5, int(fps * 0.5)) if fps > 0 else 5
    object_positions = defaultdict(lambda: deque(maxlen=object_positions_maxlen))
    
    # 객체 속도 벡터 저장소 추가 (항공뷰 모드 예측 추적용)
    object_velocity_vectors = defaultdict(lambda: deque(maxlen=3))

    object_speeds = {}
    speed_records = []
    track_class = {}
    presence_data = []
    distance_deceleration_records = []
    class_counts_O = defaultdict(int)
    unique_ids_per_class = defaultdict(set)
    id_crossed_initial_line = defaultdict(bool)
    id_stepped_tracking_line = defaultdict(bool)

    # 성능 최적화: 메모리 사용량 제한
    num_frames_smoothing = int(fps * 1) if fps > 0 else 5
    max_speed_history = int(fps * 2) if fps > 0 else 10  # 최대 2초 기록
    speed_smoothing = defaultdict(lambda: deque(maxlen=num_frames_smoothing))
    speed_history = defaultdict(lambda: deque(maxlen=max_speed_history))

    region_tracking_counts = {}

    def assign_class_to_track(tracks, detections_list):
        for tr in tracks:
            tx1, ty1, tx2, ty2, tid = map(int, tr)
            t_center = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)
            min_dist_val = float('inf')
            chosen_class = None
            for det_item in detections_list:
                dx1, dy1, dx2, dy2, conf, cls_idx = det_item
                d_center = ((dx1 + dx2) / 2, (dy1 + dy2) / 2)
                d_val = dist(t_center, d_center)
                if d_val < min_dist_val:
                    min_dist_val = d_val
                    # .get()을 사용하여 class_names가 딕셔너리일 때 안전하게 접근
                    original_class = class_names.get(int(cls_idx), "Unknown")
                    chosen_class = map_class_name(original_class) if original_class != "Unknown" else "Unknown"

            if chosen_class is not None and original_class in desired_classes and tid not in track_class:
                track_class[tid] = chosen_class
                unique_ids_per_class[chosen_class].add(tid)


    def is_point_near_line(point, start, end, threshold=10):
        x0, y0 = point
        x1, y1 = start
        x2, y2 = end

        line_seg_length_sq = (x2 - x1)**2 + (y2 - y1)**2
        if line_seg_length_sq == 0:
            return dist(point, start) < threshold

        numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt(line_seg_length_sq)
        distance_to_line = numerator / denominator

        if distance_to_line >= threshold : return False

        dot_product = (x0 - x1)*(x2 - x1) + (y0 - y1)*(y2 - y1)
        if dot_product < 0: return dist(point, start) < threshold

        dot_product2 = (x0 - x2)*(x1 - x2) + (y0 - y2)*(y1 - y2)
        if dot_product2 < 0: return dist(point, end) < threshold

        return True


    cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('YOLO Detection', 1920, 1080) # 필요에 따라 주석 해제
    cv2.setMouseCallback('YOLO Detection', yolo_mouse_callback)

    poly_pts_src = np.array(rect_points, dtype=np.float32)

    TARGET_WIDTH = width_m
    TARGET_HEIGHT = height_m
    TARGET = np.array(
        [[0, 0], [TARGET_WIDTH, 0], [TARGET_WIDTH, TARGET_HEIGHT], [0, TARGET_HEIGHT]],
        dtype=np.float32
    )
    view_transformer = ViewTransformer(source=poly_pts_src, target=TARGET)

    estimated_vertical_m = height_m
    estimated_horizontal_m = width_m
    print(f"Estimated vertical dimension: {estimated_vertical_m:.2f} m")
    print(f"Estimated horizontal dimension: {estimated_horizontal_m:.2f} m")
    
    # 항공뷰 모드 설정 안내
    print("\n" + "="*50)
    print("항공뷰 모드 사용법:")
    print("- 'a' 키: 항공뷰 모드 ON/OFF")
    print("- 항공뷰 모드에서 마우스로 기준선 그리기")
    print("- 기준선의 실제 거리 입력하여 스케일 설정")
    print("- 'r' 키: 항공뷰 스케일 초기화")
    print("="*50)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    current_frame = 0
    
    # 진행률 표시를 위한 변수
    last_progress_time = time.time()
    
    while cap.isOpened() and current_frame < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 현재 시간과 총 시간을 HH:MM:SS 형식으로 표시하는 부분 추가
        current_time_sec = current_frame / fps if fps > 0 else 0
        total_time_sec = total_frames / fps if fps > 0 else 0
        
        # 시간 포맷 변환 함수
        def format_time(seconds):
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        current_time_formatted = format_time(current_time_sec)
        total_time_formatted = format_time(total_time_sec)
        timestamp_text = f"Video Time: {current_time_formatted}/{total_time_formatted}"
        
        # 검정색 배경의 반투명 상자에 흰색 텍스트로 시간 표시
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, timestamp_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (255, 255, 255), 2)

        # YOLO 예측 - 강화된 예외 처리
        try:
            results = yolo_model.predict(source=frame, 
                                       imgsz=Constants.DEFAULT_IMGSZ, 
                                       conf=Constants.CONFIDENCE_THRESHOLD, 
                                       iou=Constants.IOU_THRESHOLD)
        except (RuntimeError, MemoryError) as e:
            print(f"YOLO 예측 오류 (프레임 {current_frame}): {e}")
            results = []
        except Exception as e:
            print(f"YOLO 예상치 못한 오류 (프레임 {current_frame}): {e}")
            results = []
        detections = []
        for result_item in results:
            boxes = result_item.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                original_cls_name = class_names.get(cls_idx, "Unknown") # .get() 사용
                # 항공뷰용 더 관대한 필터링
                if conf < Constants.FILTER_THRESHOLD or original_cls_name not in desired_classes:
                    continue
                detections.append([x1, y1, x2, y2, conf, cls_idx])
        
        # 주행모드에서의 표지판 검지 처리
        if driving_mode and len(detections) > 0:
            current_timestamp = datetime.now()
            process_sign_detection(frame, detections, current_timestamp, current_frame)

        # 주행모드에서는 추적기 생략, 다른 모드에서는 추적기 사용
        tracked_objects = np.empty((0, 5))  # 기본값: [x1, y1, x2, y2, track_id] 형식
        
        if not driving_mode:
            # 강화된 예외 처리 및 DeepSORT 호환 구현
            try:
                if len(detections) > 0:
                    # DeepSORT requires detections as list of tuples: ([left, top, w, h], confidence, class)
                    deepsort_detections = []
                    for d in detections:
                        x1, y1, x2, y2, conf, cls_idx = d
                        # Convert [x1, y1, x2, y2] to [left, top, w, h]
                        left, top, w, h = x1, y1, x2 - x1, y2 - y1
                        original_cls_name = class_names.get(int(cls_idx), "unknown") if class_names else "unknown"
                        deepsort_detections.append(([left, top, w, h], conf, original_cls_name))
                    
                    tracks = tracker.update_tracks(deepsort_detections, frame=frame)
                else:
                    # 빈 검출에서도 추적기 업데이트 (기존 트랙 유지를 위해)
                    tracks = tracker.update_tracks([], frame=frame)
            
                # Convert DeepSORT tracks to SORT-compatible format [x1, y1, x2, y2, track_id]
                tracked_objects_list = []
                
                # 디버깅 정보
                if current_frame % 100 == 0:  # 100프레임마다 로그
                    print(f"프레임 {current_frame}: {len(tracks)}개 트랙 처리 중")
                
                for track in tracks:
                    # 트랙 확인 상태 검증 (DeepSORT에서는 초기에 tentative 상태일 수 있음)
                    if not hasattr(track, 'is_confirmed'):
                        continue
                    
                    # 확인된 트랙만 사용 (n_init 번의 연속 감지 후)
                    if not track.is_confirmed():
                        continue
                        
                    # 트랙 ID 검증
                    if not hasattr(track, 'track_id'):
                        continue
                        
                    # bbox 추출 및 검증
                    try:
                        # DeepSORT의 to_ltrb() 메소드 사용 (left, top, right, bottom)
                        bbox = track.to_ltrb()
                        
                        # bbox 유효성 검증
                        if bbox is None or len(bbox) != 4:
                            continue
                            
                        # 유효한 좌표값 검증
                        x1, y1, x2, y2 = bbox
                        if not all(isinstance(coord, (int, float, np.number)) for coord in bbox):
                            continue
                            
                        # bbox 크기 검증
                        if x2 <= x1 or y2 <= y1:
                            continue
                            
                        tracked_objects_list.append([
                            float(x1), float(y1), float(x2), float(y2), int(track.track_id)
                        ])
                        
                    except (AttributeError, TypeError, ValueError) as e:
                        print(f"트랙 {getattr(track, 'track_id', 'Unknown')} bbox 추출 오류: {e}")
                        continue
                
                # 결과 배열 생성
                if tracked_objects_list:
                    tracked_objects = np.array(tracked_objects_list, dtype=np.float32)
                else:
                    tracked_objects = np.empty((0, 5), dtype=np.float32)
                    
            except (ValueError, RuntimeError, MemoryError) as e:
                print(f"추적 오류 (프레임 {current_frame}): {e}")
                tracked_objects = np.empty((0, 5), dtype=np.float32)
            except Exception as e:
                print(f"예상치 못한 추적 오류 (프레임 {current_frame}): {e}")
                print(f"오류 타입: {type(e).__name__}")
                tracked_objects = np.empty((0, 5), dtype=np.float32)

        # 주행모드가 아닐 때만 클래스 할당 수행
        if (not driving_mode and
            isinstance(tracked_objects, np.ndarray) and 
            tracked_objects.size > 0 and 
            len(tracked_objects.shape) == 2 and 
            tracked_objects.shape[1] == 5 and 
            len(detections) > 0):
            assign_class_to_track(tracked_objects, detections)

        current_time = current_frame / fps if fps > 0 else 0

        if display_info: # 'p' 키로 토글되는 정보
            if not driving_mode:  # 주행모드가 아닐 때만 거리 정보 표시
                cv2.putText(frame, f"Est. Vert: {estimated_vertical_m:.2f}m Horiz: {estimated_horizontal_m:.2f}m", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            # 주행모드와 항공뷰 모드가 아닐 때만 ROI 박스 표시
            if not aerial_view_mode and not driving_mode and len(rect_points) == 4:
                pts_roi = np.array(rect_points, dtype=np.int32)
                cv2.polylines(frame, [pts_roi], True, (0,255,0), 2)


        current_tracked_ids = set()
        
        # tracked_objects 유효성 검증 후 반복 처리
        if (isinstance(tracked_objects, np.ndarray) and 
            tracked_objects.size > 0 and 
            len(tracked_objects.shape) == 2 and 
            tracked_objects.shape[1] == 5):
            
            for trk_item in tracked_objects:
                try:
                    # 트랙 데이터 언패킹 및 검증
                    if not isinstance(trk_item, (np.ndarray, list)) or len(trk_item) != 5:
                        continue
                    x1_trk, y1_trk, x2_trk, y2_trk, track_id = map(float, trk_item)
                    track_id = int(track_id)
                    
                    # 유효한 bbox인지 검증
                    if x2_trk <= x1_trk or y2_trk <= y1_trk:
                        continue
                        
                    label = f"ID {track_id}"

                    center_x = (x1_trk + x2_trk) / 2
                    center_y = (y1_trk + y2_trk) / 2
                    bbox_y2 = y2_trk

                    object_positions[track_id].append((center_x, center_y, bbox_y2, current_time))
                    current_tracked_ids.add(track_id)
                    
                    # 누적 trail 경로에 현재 위치 저장 (항상 백그라운드에서 수집)
                    if track_id not in accumulated_trail_paths:
                        accumulated_trail_paths[track_id] = []
                    accumulated_trail_paths[track_id].append((center_x, center_y))

                    xs_list = [p[0] for p in object_positions[track_id]]
                    ys_center_list = [p[1] for p in object_positions[track_id]]

                    # 항공뷰 모드에서 더 강력한 실시간 추적 적용
                    if aerial_view_mode and len(object_positions[track_id]) >= 2:
                        # 최근 2-3개 위치만 사용하여 지연 최소화
                        recent_positions = list(object_positions[track_id])[-3:]
                        recent_xs = [p[0] for p in recent_positions]
                        recent_ys = [p[1] for p in recent_positions]
                        
                        # 최신 위치에 매우 높은 가중치 적용 (0.7~1.0)
                        weights = np.linspace(0.7, 1.0, len(recent_xs))
                        smooth_x = np.average(recent_xs, weights=weights)
                        smooth_y_center = np.average(recent_ys, weights=weights)
                        
                        # 더 강력한 예측: 최근 2프레임 속도로 더 멀리 예측
                        if len(recent_positions) >= 2:
                            current_pos = recent_positions[-1]
                            prev_pos = recent_positions[-2]
                            dt = max(0.016, current_pos[3] - prev_pos[3])  # 최소 16ms (60fps)
                            
                            vx = (current_pos[0] - prev_pos[0]) / dt
                            vy = (current_pos[1] - prev_pos[1]) / dt
                            object_velocity_vectors[track_id].append((vx, vy))
                            
                            # 더 긴 예측 시간으로 지연 보상 강화
                            prediction_time = 0.1  # 100ms 앞 예측
                            smooth_x += vx * prediction_time
                            smooth_y_center += vy * prediction_time
                    else:
                        # 호모그래피 모드: 기존 단순 평균 사용
                        smooth_x = np.mean(xs_list) if xs_list else center_x
                        smooth_y_center = np.mean(ys_center_list) if ys_center_list else center_y

                    speed_kph = None
                    acceleration_mps2 = None
                    deceleration_mps2 = None
                    acceleration_percent = None
                    deceleration_percent = None

                    if len(object_positions[track_id]) >= 2 and len(object_positions[track_id]) >= object_positions_maxlen // 2:
                        curr_pos_data = object_positions[track_id][-1]
                        prev_pos_data = object_positions[track_id][0]

                        dt_interval = curr_pos_data[3] - prev_pos_data[3]

                        if dt_interval > 0:
                            # 항공뷰 모드 또는 기존 호모그래피 모드에 따라 거리 계산 방식 선택
                            if aerial_view_mode and aerial_meters_per_pixel is not None:
                                # 항공뷰 모드: 선형 스케일 기반 거리 계산
                                current_pixel_pos = (curr_pos_data[0], curr_pos_data[2])
                                prev_pixel_pos = (prev_pos_data[0], prev_pos_data[2])
                                dist_m_val = calculate_aerial_distance(prev_pixel_pos, current_pixel_pos)
                            else:
                                # 기존 호모그래피 모드
                                real_pos_current = view_transformer.transform_points(np.array([[curr_pos_data[0], curr_pos_data[2]]]))[0]
                                real_pos_prev = view_transformer.transform_points(np.array([[prev_pos_data[0], prev_pos_data[2]]]))[0]
                                dist_m_val = dist(real_pos_current, real_pos_prev)
                            
                            if dist_m_val is not None and dist_m_val > 0:
                                speed_mps = dist_m_val / dt_interval
                            else:
                                speed_mps = 0

                            speed_smoothing[track_id].append(speed_mps)
                            avg_speed_mps = np.mean(speed_smoothing[track_id]) if speed_smoothing[track_id] else speed_mps

                            object_speeds[track_id] = avg_speed_mps
                            speed_kph = avg_speed_mps * 3.6

                            speed_history[track_id].append((avg_speed_mps, current_time))
                            while speed_history[track_id] and (current_time - speed_history[track_id][0][1]) > 0.5:
                                speed_history[track_id].popleft()

                            if len(speed_history[track_id]) >= 2:
                                current_speed_data = speed_history[track_id][-1]
                                first_speed_data_in_window = speed_history[track_id][0]

                                delta_v = current_speed_data[0] - first_speed_data_in_window[0]
                                delta_t_accel = current_speed_data[1] - first_speed_data_in_window[1]

                                if delta_t_accel > 0:
                                    calculated_accel_mps2 = delta_v / delta_t_accel
                                    if abs(calculated_accel_mps2) > Constants.ACCEL_LIMIT_MPS2:
                                        calculated_accel_mps2 = None

                                    if calculated_accel_mps2 is not None:
                                        base_speed_for_percent = first_speed_data_in_window[0] + 1e-5
                                        if calculated_accel_mps2 < 0:
                                            deceleration_mps2 = abs(calculated_accel_mps2)
                                            deceleration_percent = (deceleration_mps2 / base_speed_for_percent) * 100 if base_speed_for_percent != 0 else 0
                                        else:
                                            acceleration_mps2 = calculated_accel_mps2
                                            acceleration_percent = (acceleration_mps2 / base_speed_for_percent) * 100 if base_speed_for_percent != 0 else 0

                    cls_name_trk = track_class.get(track_id, "Unknown")
                    # 클래스 이름이 이미 매핑되어 저장됨 (track_class에 저장할 때 map_class_name 적용됨)

                    # 차량 타입별 색상 처리
                    if cls_name_trk == 'CAR-SUV':
                        circle_color = (0, 255, 255)  # 노란색 원 (VAN -> CAR-SUV)
                        text_color = (0, 255, 255)    # 노란색 텍스트
                    elif cls_name_trk == 'CAR':
                        circle_color = (0, 255, 0)    # 초록색 원
                        text_color = (0, 255, 0)      # 초록색 텍스트
                    elif cls_name_trk == 'TRUCK':
                        circle_color = (255, 0, 0)    # 파란색 원
                        text_color = (255, 0, 0)      # 파란색 텍스트
                    elif cls_name_trk.upper() in ['DRONE', 'UAV', 'AIRCRAFT', 'HELICOPTER']:
                        circle_color = (255, 0, 255)  # 보라색 원
                        text_color = (255, 0, 255)    # 보라색 텍스트
                    else:
                        circle_color = (0, 0, 255)    # 기본 빨간색 원
                        text_color = (0, 255, 0)      # 기본 초록색 텍스트

                    # 객체 이동 경로 표시 (trail 기능)
                    if show_trails and len(object_positions[track_id]) > 1:
                        trail_positions = list(object_positions[track_id])
                        
                        # trail 색상 설정 (ID별로 다른 색상)
                        trail_colors = [
                            (255, 100, 100),  # 연한 빨강
                            (100, 255, 100),  # 연한 초록
                            (100, 100, 255),  # 연한 파랑
                            (255, 255, 100),  # 연한 노랑
                            (255, 100, 255),  # 연한 자홍
                            (100, 255, 255),  # 연한 시안
                        ]
                        trail_color = trail_colors[track_id % len(trail_colors)]
                        
                        # 이동 경로를 선으로 연결
                        for i in range(1, len(trail_positions)):
                            pt1 = (int(trail_positions[i-1][0]), int(trail_positions[i-1][1]))
                            pt2 = (int(trail_positions[i][0]), int(trail_positions[i][1]))
                            cv2.line(frame, pt1, pt2, trail_color, 2)
                    
                    # 추적점 표시
                    if aerial_view_mode:
                        # 항공뷰 모드: 더 큰 추적점
                        cv2.circle(frame, (int(smooth_x), int(smooth_y_center)), 7, circle_color, -1)
                        cv2.circle(frame, (int(smooth_x), int(smooth_y_center)), 10, circle_color, 2)  # 바깥쪽 테두리
                    else:
                        # 호모그래피 모드: 기존 표시 방식
                        cv2.circle(frame, (int(smooth_x), int(smooth_y_center)), 5, circle_color, -1)

                    info_label = f"{label} {cls_name_trk}"
                    cv2.putText(frame, info_label, (int(x1_trk), int(y1_trk) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

                    if speed_kph is not None:
                        cv2.putText(frame, f"{speed_kph:.1f}km/h", (int(x1_trk), int(y1_trk) - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                        speed_records.append({
                            "Frame": current_frame, "Time": current_time, "ID": track_id, "Class": cls_name_trk,
                            "IsInitialLine": id_crossed_initial_line[track_id],
                            "IsSteppedTrackingLine": id_stepped_tracking_line[track_id],
                            "Speed(km/h)": speed_kph,
                            "Acceleration(m/s²)": acceleration_mps2, "Deceleration(m/s²)": deceleration_mps2,
                            "Acceleration (%)": acceleration_percent, "Deceleration (%)": deceleration_percent
                        })

                    presence_data.append({
                        "Frame": current_frame, "Time": current_time, "ID": track_id, "Class": cls_name_trk,
                        "IsInitialLine": id_crossed_initial_line[track_id],
                        "IsSteppedTrackingLine": id_stepped_tracking_line[track_id]
                    })

                    for line_data in tracking_lines:
                        if len(line_data["points"]) == 2:
                            start_pt, end_pt = line_data["points"]
                            point_for_line_test = (smooth_x, smooth_y_center)

                            if is_point_near_line(point_for_line_test, start_pt, end_pt) and \
                               track_id not in line_data.get("persistent_crossed_ids", set()):

                                line_data.setdefault("crossed_ids", set()).add(track_id)
                                line_data.setdefault("persistent_crossed_ids", set()).add(track_id)
                                line_data["count_cross"] = len(line_data["persistent_crossed_ids"])

                                if line_data["type"] == "initial":
                                    id_crossed_initial_line[track_id] = True
                                    if cls_name_trk != "Unknown":
                                        class_counts_O[cls_name_trk] += 1
                                    crossing_data.append({
                                        "ID": track_id, "Class": cls_name_trk, "Time": current_time,
                                        "Line": line_data["name"], "IsInitialLine": True
                                    })
                                else:
                                    id_stepped_tracking_line[track_id] = True
                                    crossing_data.append({
                                        "ID": track_id, "Class": cls_name_trk, "Time": current_time,
                                        "Line": line_data["name"], "IsInitialLine": id_crossed_initial_line[track_id]
                                    })

                                line_data.setdefault("display_crossed_ids", set()).add(track_id)

                    center_for_region_test = (smooth_x, smooth_y_center)
                    for idx, region_poly in enumerate(custom_regions):
                        if cv2.pointPolygonTest(region_poly, center_for_region_test, False) >= 0:
                            stepped_any_line = id_stepped_tracking_line[track_id]
                            lines_stepped_names = [ln["name"] for ln in tracking_lines
                                                   if track_id in ln.get("crossed_ids", set()) and ln["type"] == "tracking"]

                            region_records.append({
                                "Region": idx + 1, "Frame": current_frame, "Time": current_time,
                                "ID": track_id, "Class": cls_name_trk,
                                "SteppedTrackingLine": stepped_any_line,
                                "Speed(km/h)": speed_kph if speed_kph is not None else 0,
                                "Acceleration(m/s²)": acceleration_mps2 if acceleration_mps2 is not None else None,
                                "Deceleration(m/s²)": deceleration_mps2 if deceleration_mps2 is not None else None,
                                "LinesStepped": ";".join(lines_stepped_names)
                            })
                            region_unique_ids[idx].add(track_id)

                except (ValueError, TypeError, IndexError) as e:
                    print(f"트랙 {track_id} 처리 중 오류: {e}")
                    continue

        # tracked_objects 유효성 재검증 (안전한 접근)
        try:
            if (isinstance(tracked_objects, np.ndarray) and 
                tracked_objects.size > 0 and 
                len(tracked_objects.shape) == 2 and 
                tracked_objects.shape[1] == 5):
                tracked_ids_in_current_frame = set([int(trk[4]) for trk in tracked_objects])
            else:
                tracked_ids_in_current_frame = set()
        except (ValueError, TypeError, IndexError):
            tracked_ids_in_current_frame = set()
        all_known_ids = set(object_positions.keys())
        disappeared_ids = all_known_ids - tracked_ids_in_current_frame

        for d_id in disappeared_ids:
            if d_id in object_positions: del object_positions[d_id]
            if d_id in object_speeds: del object_speeds[d_id]
            if d_id in track_class: del track_class[d_id]
            if d_id in speed_smoothing: del speed_smoothing[d_id]
            if d_id in speed_history: del speed_history[d_id]
            if d_id in object_velocity_vectors: del object_velocity_vectors[d_id]  # 속도 벡터도 정리
            for line_data_item in tracking_lines:
                line_data_item.get("crossed_ids", set()).discard(d_id)
                line_data_item.get("display_crossed_ids", set()).discard(d_id)

        # 성능 최적화: 차간 거리 계산 개선
        def calculate_vehicle_distances():
            """차간 거리 계산을 최적화된 방식으로 처리"""
            # 최근 속도 데이터를 딕셔너리로 캐싱 (O(1) 접근)
            latest_speed_data = {}
            for rec in reversed(speed_records[-Constants.SPEED_CACHE_SIZE:]):  # 최근 N개만 확인
                vehicle_id = rec["ID"]
                if vehicle_id not in latest_speed_data:
                    latest_speed_data[vehicle_id] = {
                        "Acceleration(m/s²)": rec["Acceleration(m/s²)"],
                        "Deceleration(m/s²)": rec["Deceleration(m/s²)"],
                        "Acceleration (%)": rec["Acceleration (%)"],
                        "Deceleration (%)": rec["Deceleration (%)"]
                    }
            
            for line_data_dist in tracking_lines:
                if line_data_dist["type"] != "tracking": 
                    continue

                active_ids_on_line = list(line_data_dist.get("persistent_crossed_ids", set()).intersection(tracked_ids_in_current_frame))
                
                if len(active_ids_on_line) < 2: 
                    continue

                # 위치 데이터 한 번에 수집
                valid_positions = []
                for vehicle_id in active_ids_on_line:
                    if object_positions[vehicle_id]:
                        pos_data = object_positions[vehicle_id][-1]
                        valid_positions.append((vehicle_id, pos_data))
                
                # 모든 쌍 처리 (개선된 방식)
                from itertools import combinations
                for (id1, pos1), (id2, pos2) in combinations(valid_positions, 2):
                    p1_pixel = (pos1[0], pos1[1])
                    p2_pixel = (pos2[0], pos2[1])

                    # 변환 포인트 계산
                    transform_points = np.array([[pos1[0], pos1[2]], [pos2[0], pos2[2]]])
                    real_positions = view_transformer.transform_points(transform_points)
                    distance_m_val2 = dist(real_positions[0], real_positions[1])

                    # 속도 정보
                    speed1_kph = object_speeds.get(id1, 0) * 3.6
                    speed2_kph = object_speeds.get(id2, 0) * 3.6

                    # 가속도 정보 (캐시된 데이터 사용)
                    data1 = latest_speed_data.get(id1, {})
                    data2 = latest_speed_data.get(id2, {})

                    distance_deceleration_records.append({
                        "Frame": current_frame, "Time": current_time,
                        "ID1": id1, "Class1": track_class.get(id1, "Unknown"), "Speed1(km/h)": speed1_kph,
                        "Acceleration1(m/s²)": data1.get("Acceleration(m/s²)"), 
                        "Deceleration1(m/s²)": data1.get("Deceleration(m/s²)"),
                        "Acceleration1 (%)": data1.get("Acceleration (%)"), 
                        "Deceleration1 (%)": data1.get("Deceleration (%)"),
                        "ID2": id2, "Class2": track_class.get(id2, "Unknown"), "Speed2(km/h)": speed2_kph,
                        "Acceleration2(m/s²)": data2.get("Acceleration(m/s²)"), 
                        "Deceleration2(m/s²)": data2.get("Deceleration(m/s²)"),
                        "Acceleration2 (%)": data2.get("Acceleration (%)"), 
                        "Deceleration2 (%)": data2.get("Deceleration (%)"),
                        "Distance(m)": distance_m_val2, "Line": line_data_dist["name"],
                        "IsInitialLine_ID1": id_crossed_initial_line[id1],
                        "IsInitialLine_ID2": id_crossed_initial_line[id2]
                    })
                    
                    # 화면에 거리 표시
                    mid_point_pixel = ((p1_pixel[0] + p2_pixel[0]) / 2, (p1_pixel[1] + p2_pixel[1]) / 2)
                    cv2.putText(frame, f"{distance_m_val2:.1f}m", (int(mid_point_pixel[0]), int(mid_point_pixel[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
                    cv2.line(frame, (int(p1_pixel[0]), int(p1_pixel[1])), (int(p2_pixel[0]), int(p2_pixel[1])), (255,0,255), 1)

        # 매 프레임마다 실행하지 말고 N프레임마다 실행 (성능 향상)
        if current_frame % Constants.DISTANCE_CALC_INTERVAL == 0:
            calculate_vehicle_distances()


        # 누적 trail 표시 (불투명 색상 스택으로 차로 사용량 분석)
        if show_accumulated_trails:
            # 불투명 overlay 생성
            overlay = frame.copy()
            
            for obj_id, path in accumulated_trail_paths.items():
                if len(path) > 1:
                    # 불투명 색상들 (겹칠수록 진해짐)
                    trail_colors = [
                        (50, 150, 255),   # 연한 주황
                        (50, 255, 150),   # 연한 초록
                        (255, 150, 50),   # 연한 파랑
                        (150, 255, 255),  # 연한 노랑
                        (255, 50, 255),   # 연한 자홍
                        (255, 255, 50),   # 연한 시안
                        (180, 50, 180),   # 연한 보라
                        (255, 200, 50),   # 연한 진한 주황
                    ]
                    
                    # 객체 ID에 따른 색상 선택
                    color = trail_colors[obj_id % len(trail_colors)]
                    
                    # overlay에 경로 그리기 (겹칠수록 진해짐)
                    for i in range(1, len(path)):
                        pt1 = (int(path[i-1][0]), int(path[i-1][1]))
                        pt2 = (int(path[i][0]), int(path[i][1]))
                        cv2.line(overlay, pt1, pt2, color, 5)  # 더 두꺼운 선
            
            # 불투명도 적용하여 원본 프레임과 블렌딩 (겹칠수록 진해지는 효과)
            alpha = 0.3  # 불투명도 (0.0 = 완전 투명, 1.0 = 완전 불투명)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # tracking line 그리기 개선 (디버깅 정보 포함)
        # 주행모드가 아닐 때만 tracking_lines 그리기
        if not driving_mode:
            if len(tracking_lines) > 0:
                print(f"[화면 그리기] tracking_lines 개수: {len(tracking_lines)}")
            
            for line_data_draw in tracking_lines:
                line_points = line_data_draw["points"]
                line_type = line_data_draw["type"]
                line_name = line_data_draw["name"]
                
                # 선 색상: 초기선은 녹색, 추적선은 빨간색
                color = (0, 255, 0) if line_type == "initial" else (0, 0, 255)
                
                if len(line_points) == 2:
                    # 완성된 선 그리기
                    start_draw, end_draw = line_points
                    cv2.line(frame, start_draw, end_draw, color, 3)  # 선 두께 증가
                    actual_count = len(line_data_draw.get("persistent_crossed_ids", set()))
                    
                    # 텍스트 배경 추가로 가독성 향상
                    label = f"{line_name} Cnt: {actual_count}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame, 
                                 (int(start_draw[0]), int(start_draw[1])-25),
                                 (int(start_draw[0]) + label_size[0], int(start_draw[1])-5),
                                 (0, 0, 0), -1)
                    cv2.putText(frame, label, (int(start_draw[0]), int(start_draw[1])-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                               
                elif len(line_points) == 1:
                    # 미완성 선 (첫 번째 점만 있는 경우) - 시작점 표시
                    start_point = line_points[0]
                    cv2.circle(frame, start_point, 8, color, -1)
                    cv2.putText(frame, f"{line_name} (미완성)", 
                               (int(start_point[0]) + 10, int(start_point[1]) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 모든 점 표시 (드래그 가능한 점들)
                for i, p_vis in enumerate(line_points):
                    cv2.circle(frame, p_vis, 5, (255, 0, 0), -1)  # 파란색 원
                    cv2.circle(frame, p_vis, 6, (255, 255, 255), 1)  # 흰색 테두리

        # 모드별 화면 표시
        if driving_mode:
            # 주행모드 상태 표시
            mode_text = "Driving Mode: Sign Detection"
            cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 현재 검지된 표지판 수 표시
            total_detections = len(sign_detection_data) if 'sign_detection_data' in locals() else 0
            detection_text = f"Signs Detected: {total_detections}"
            cv2.putText(frame, detection_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        elif aerial_view_mode:
            # 항공뷰 모드 상태 표시 (기준선은 이미 설정 완료되어 표시하지 않음)
            mode_text = "Aerial View Mode: ON"
            if aerial_meters_per_pixel is not None:
                mode_text += f" (Scale: {aerial_meters_per_pixel:.4f} m/px)"
            else:
                mode_text += " - Scale not set"
                
            cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 항공뷰 모드에서 추적 성능 정보 표시
            if aerial_view_mode:
                # 추적 중인 객체 수와 스무딩 설정 정보
                tracking_info = f"Tracking: {len(current_tracked_ids)} objects | Smoothing: Reduced for accuracy"
                cv2.putText(frame, tracking_info, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 예측 기반 추적 활성화 표시
                cv2.putText(frame, "Predictive tracking: ENABLED", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Aerial View Mode: OFF (Press 'a' to enable)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # 마우스 사용법 도움말 표시
        if not custom_region_mode:
            help_lines = [
                "Mouse Controls:",
                "Right Click: Add/Complete Tracking Line (Red)",
                "Wheel Click: Add/Complete Initial Line (Green)", 
                "Left Click: Drag line endpoints",
                "Press 'c': Custom region mode"
            ]
            for i, help_line in enumerate(help_lines):
                y_pos = frame.shape[0] - 120 + (i * 20)  # 화면 하단에서 위로
                color = (255, 255, 0) if i == 0 else (255, 255, 255)  # 제목은 노란색
                thickness = 2 if i == 0 else 1
                cv2.putText(frame, help_line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        # 'p' 키 토글 정보는 위로 이동

        text_x_custom = width - 400 if width > 400 else 20 # 화면 너비에 따라 조정
        y_offset_custom = 20
        for idx_cr, region_cr in enumerate(custom_regions):
            cv2.polylines(frame, [region_cr], True, (0, 255, 255), 2)

            region_speeds_list = []
            region_decel_list = []

            ids_in_region_now = []
            for trk_id_loop in tracked_ids_in_current_frame:
                if object_positions[trk_id_loop]:
                    last_pos_data = object_positions[trk_id_loop][-1]
                    center_for_custom_region_test = (last_pos_data[0], last_pos_data[1])
                    if cv2.pointPolygonTest(region_cr, center_for_custom_region_test, False) >= 0:
                        ids_in_region_now.append(trk_id_loop)
                        if trk_id_loop in object_speeds:
                            region_speeds_list.append(object_speeds[trk_id_loop] * 3.6)
                        for rec_sr in reversed(speed_records):
                            if rec_sr["ID"] == trk_id_loop and rec_sr["Deceleration(m/s²)"] is not None:
                                region_decel_list.append(rec_sr["Deceleration(m/s²)"])
                                break

            avg_speed_custom = np.mean(region_speeds_list) if region_speeds_list else 0
            avg_decel_custom = np.mean(region_decel_list) if region_decel_list else 0

            if idx_cr not in region_tracking_counts: region_tracking_counts[idx_cr] = {}
            for line_data_cr in tracking_lines:
                if len(line_data_cr["points"]) == 2:
                    p1_line, p2_line = line_data_cr["points"]
                    mid_line = ((p1_line[0] + p2_line[0]) / 2, (p1_line[1] + p2_line[1]) / 2)
                    if cv2.pointPolygonTest(region_cr, mid_line, False) >= 0:
                        line_name_cr = line_data_cr["name"]
                        current_global_count = len(line_data_cr.get("persistent_crossed_ids",set()))
                        region_tracking_counts[idx_cr][line_name_cr] = current_global_count


            region_text = (
                f"영역 {idx_cr+1}:\n"
                f"평균속도: {avg_speed_custom:.1f}km/h\n"
                f"평균감속도: {avg_decel_custom:.1f}m/s2\n"
                f"차선변경 검지선:"
            )
            for line_name_disp, count_disp in region_tracking_counts[idx_cr].items():
                region_text += f"\n  {line_name_disp}: {count_disp}"

            frame = draw_multiline_text(frame, region_text, (text_x_custom, y_offset_custom), font_size=18, color=(255,255,255))
            lines_count_rt = region_text.split('\n')
            line_height_rt = 18 + 5
            y_offset_custom += (line_height_rt * len(lines_count_rt)) + 15


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('n'):
            custom_region_mode = True
            current_custom_region.clear()
        elif key == ord('p'):
            display_info = not display_info
        elif key == ord('t'):
            # Trail 표시 토글
            show_trails = not show_trails
            print(f"객체 이동 경로 표시: {'ON' if show_trails else 'OFF'}")
        elif key == ord('i'):
            # 누적 trail 표시 토글 (차로 사용량 분석용)
            show_accumulated_trails = not show_accumulated_trails
            if show_accumulated_trails:
                print("누적 Trail 표시 ON - 모든 객체의 전체 이동 경로 표시 (차로 사용량 분석)")
                print(f"현재까지 {len(accumulated_trail_paths)}개 객체의 경로 데이터 보유")
            else:
                print("누적 Trail 표시 OFF")
        elif key == ord('a'):
            # 항공뷰 모드는 시작 시 설정되므로 분석 중 토글 불가
            if aerial_view_mode:
                print("항공뷰 모드는 이미 활성화되어 있으며, 분석 중에는 변경할 수 없습니다.")
            else:
                print("항공뷰 모드를 사용하려면 처음부터 항공뷰 모드로 분석을 시작해야 합니다.")
        elif key == ord('r'):
            # 항공뷰 스케일은 시작 시 설정되므로 재설정 불가
            if aerial_view_mode:
                print("항공뷰 스케일은 분석 시작 시 설정되었으며, 분석 중에는 변경할 수 없습니다.")
            else:
                print("호모그래피 모드에서는 스케일 재설정 기능이 없습니다.")

        # 비디오 프레임 쓰기 시도
        if out and out.isOpened():
            success = out.write(frame)
            if not success and current_frame % 100 == 0:  # 100프레임마다 한번씩 체크
                print(f"경고: 프레임 {current_frame} 쓰기 실패")
        else:
            if current_frame % 100 == 0:  # 100프레임마다 한번씩 경고
                print(f"경고: VideoWriter가 열려있지 않음 (프레임 {current_frame})")
        
        cv2.imshow('YOLO Detection', frame)
        current_frame += 1
        
        # 진행률 표시 (5초마다)
        current_time_now = time.time()
        if current_time_now - last_progress_time > 5:
            progress_pct = (current_frame / total_frames) * 100 if total_frames > 0 else 0
            print(f"진행률: {progress_pct:.1f}% ({current_frame}/{total_frames} 프레임)")
            last_progress_time = current_time_now

    cap.release()
    
    # 비디오 저장 완료 확인
    if out:
        out.release()
        # 파일이 실제로 저장되었는지 확인
        if os.path.exists(output_video_path):
            file_size = os.path.getsize(output_video_path)
            print(f"✅ 비디오 저장 완료: {output_video_path}")
            print(f"   파일 크기: {file_size:,} bytes ({file_size/(1024*1024):.1f} MB)")
        else:
            print(f"❌ 비디오 파일이 생성되지 않았습니다: {output_video_path}")
    else:
        print("❌ VideoWriter가 초기화되지 않아 비디오가 저장되지 않았습니다.")
    
    # 영상 분석 완료 후 누적 trail 스크린샷 저장
    if accumulated_trail_paths and len(accumulated_trail_paths) > 0:
        print("📸 누적 trail 스크린샷을 생성합니다...")
        
        # 마지막 프레임을 다시 열어서 trail stack 이미지 생성
        cap_screenshot = cv2.VideoCapture(input_video_path)
        if cap_screenshot.isOpened():
            # 마지막 프레임으로 이동
            cap_screenshot.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
            ret_ss, frame_ss = cap_screenshot.read()
            
            if ret_ss:
                # 누적 trail overlay 생성
                overlay_ss = frame_ss.copy()
                
                for obj_id, path in accumulated_trail_paths.items():
                    if len(path) > 1:
                        # 불투명 색상들
                        trail_colors = [
                            (50, 150, 255),   # 연한 주황
                            (50, 255, 150),   # 연한 초록
                            (255, 150, 50),   # 연한 파랑
                            (150, 255, 255),  # 연한 노랑
                            (255, 50, 255),   # 연한 자홍
                            (255, 255, 50),   # 연한 시안
                            (180, 50, 180),   # 연한 보라
                            (255, 200, 50),   # 연한 진한 주황
                        ]
                        
                        color = trail_colors[obj_id % len(trail_colors)]
                        
                        # overlay에 전체 경로 그리기
                        for i in range(1, len(path)):
                            pt1 = (int(path[i-1][0]), int(path[i-1][1]))
                            pt2 = (int(path[i][0]), int(path[i][1]))
                            cv2.line(overlay_ss, pt1, pt2, color, 5)
                
                # 불투명도 적용하여 블렌딩
                alpha = 0.4  # 스크린샷용으로 조금 더 진하게
                cv2.addWeighted(overlay_ss, alpha, frame_ss, 1 - alpha, 0, frame_ss)
                
                # 스크린샷 저장 경로
                screenshot_filename = f"{video_name_without_ext}_trail_analysis.png"
                screenshot_path = os.path.join(output_folder_path, screenshot_filename)
                
                # 스크린샷 저장
                if cv2.imwrite(screenshot_path, frame_ss):
                    print(f"✅ 누적 trail 분석 이미지 저장 완료: {screenshot_path}")
                    print(f"   분석된 객체 수: {len(accumulated_trail_paths)}개")
                else:
                    print(f"❌ 스크린샷 저장 실패: {screenshot_path}")
            
            cap_screenshot.release()
        else:
            print("❌ 스크린샷 생성을 위한 비디오 파일 열기 실패")
    else:
        print("📸 누적 trail 데이터가 없어 스크린샷을 생성하지 않습니다.")
    
    cv2.destroyAllWindows()

    # --- 주행모드 데이터 저장 ---
    if driving_mode and sign_detection_data:
        print("💾 표지판 검지 데이터를 CSV로 저장하는 중...")
        
        # 표지판 검지 데이터프레임 생성
        df_signs = pd.DataFrame(sign_detection_data)
        
        # 시간 포맷 추가
        df_signs['timestamp_formatted'] = df_signs['timestamp'].dt.strftime('%H:%M:%S.%f').str[:-3]
        df_signs['video_time_sec'] = df_signs['frame_number'] / fps if fps > 0 else 0
        df_signs['video_time_formatted'] = df_signs['video_time_sec'].apply(
            lambda x: f"{int(x//3600):02d}:{int((x%3600)//60):02d}:{int(x%60):02d}.{int((x%1)*1000):03d}"
        )
        
        # 컬럼 순서 재정렬
        sign_columns = [
            'timestamp_formatted', 'video_time_formatted', 'video_time_sec',
            'frame_number', 'class_name', 'confidence',
            'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'screenshot_path'
        ]
        df_signs = df_signs[sign_columns]
        
        # CSV 파일 저장
        signs_csv_path = os.path.join(output_folder_path, f"{current_datetime}_{clean_model_name}_{clean_video_name}_sign_detections.csv")
        df_signs.to_csv(signs_csv_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ 표지판 검지 CSV 저장 완료: {signs_csv_path}")
        print(f"   총 검지된 표지판: {len(df_signs)}개")
        print(f"   검지된 표지판 종류: {', '.join(df_signs['class_name'].unique())}")
        
        # 간단한 통계 정보 생성
        sign_summary = df_signs['class_name'].value_counts().to_dict()
        summary_text = []
        summary_text.append("=== 표지판 검지 요약 ===")
        summary_text.append(f"총 검지 수: {len(df_signs)}개")
        summary_text.append("표지판별 검지 수:")
        for sign_type, count in sign_summary.items():
            summary_text.append(f"  - {sign_type}: {count}개")
        summary_text.append(f"분석 영상: {video_name}")
        summary_text.append(f"사용 모델: {model_name}")
        
        # 요약 정보를 텍스트 파일로 저장
        summary_path = os.path.join(output_folder_path, f"{current_datetime}_{clean_model_name}_{clean_video_name}_sign_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_text))
        
        print(f"✅ 표지판 검지 요약 저장 완료: {summary_path}")
        
        # 가려짐 이벤트 CSV 저장
        if occlusion_events:
            df_occlusion = pd.DataFrame(occlusion_events)
            
            # 시간 포맷 추가
            df_occlusion['timestamp_formatted'] = df_occlusion['timestamp'].dt.strftime('%H:%M:%S.%f').str[:-3]
            df_occlusion['video_time_sec'] = df_occlusion['frame_number'] / fps if fps > 0 else 0
            df_occlusion['video_time_formatted'] = df_occlusion['video_time_sec'].apply(
                lambda x: f"{int(x//3600):02d}:{int((x%3600)//60):02d}:{int(x%60):02d}.{int((x%1)*1000):03d}"
            )
            
            # 바운딩 박스 정보 분리
            df_occlusion[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']] = pd.DataFrame(
                df_occlusion['bbox'].tolist(), index=df_occlusion.index)
            df_occlusion = df_occlusion.drop('bbox', axis=1)
            
            # 컬럼 순서 재정렬
            occlusion_columns = [
                'timestamp_formatted', 'video_time_formatted', 'video_time_sec',
                'frame_number', 'track_id', 'class_name', 'occ_ratio', 'severity', 'method',
                'is_edge_cut', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'
            ]
            df_occlusion = df_occlusion[occlusion_columns]
            
            # 가려짐 CSV 저장
            occlusion_csv_path = os.path.join(output_folder_path, 
                f"{current_datetime}_{clean_model_name}_{clean_video_name}_occlusion_events.csv")
            df_occlusion.to_csv(occlusion_csv_path, index=False, encoding='utf-8-sig')
            
            print(f"✅ 가려짐 이벤트 CSV 저장 완료: {occlusion_csv_path}")
            print(f"   총 가려짐 이벤트: {len(df_occlusion)}개")
        
        # 결과 폴더 자동으로 열기
        try:
            if os.name == 'nt':  # Windows
                os.startfile(output_folder_path)
            elif os.name == 'posix':  # macOS/Linux
                subprocess.run(['open', output_folder_path], check=True)
            print(f"📂 결과 폴더를 열었습니다: {output_folder_path}")
        except Exception as e:
            print(f"⚠️ 폴더 열기 실패: {e}")
        
        # 주행모드에서는 여기서 종료
        messagebox.showinfo("분석 완료", 
                           f"표지판 검지 분석이 완료되었습니다!\n\n"
                           f"결과 저장 위치: {output_folder_path}\n"
                           f"- 표지판 검지 CSV: {len(df_signs)}개 기록\n"
                           f"- 표지판 스크린샷: {len([f for f in os.listdir(output_folder_path) if f.startswith('sign_')])}개\n"
                           f"- 가려짐 이벤트 CSV: {len(occlusion_events)}개 기록\n"
                           f"- 분석 영상\n"
                           f"- 검지 요약 파일\n\n"
                           f"📂 결과 폴더가 자동으로 열렸습니다!")
        return
    elif driving_mode:
        # 표지판이 검지되지 않았어도 빈 CSV와 요약 파일 생성
        print("⚠️ 주행모드에서 표지판이 검지되지 않았습니다. 빈 결과 파일을 생성합니다.")
        
        # 빈 데이터프레임 생성
        empty_sign_columns = [
            'timestamp_formatted', 'video_time_formatted', 'video_time_sec',
            'frame_number', 'class_name', 'confidence',
            'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'screenshot_path'
        ]
        df_signs_empty = pd.DataFrame(columns=empty_sign_columns)
        
        # 빈 CSV 파일 저장
        signs_csv_path = os.path.join(output_folder_path, f"{current_datetime}_{clean_model_name}_{clean_video_name}_sign_detections.csv")
        df_signs_empty.to_csv(signs_csv_path, index=False, encoding='utf-8-sig')
        
        # 빈 요약 정보 생성
        summary_text = []
        summary_text.append("=== 표지판 검지 요약 ===")
        summary_text.append("총 검지 수: 0개")
        summary_text.append("검지된 표지판: 없음")
        summary_text.append(f"분석 영상: {video_name}")
        summary_text.append(f"사용 모델: {model_name}")
        summary_text.append("\n※ 표지판이 검지되지 않았습니다.")
        summary_text.append("- 모델이 표지판 검지용으로 훈련되었는지 확인하세요.")
        summary_text.append("- 영상에 표지판이 포함되어 있는지 확인하세요.")
        summary_text.append("- 검지 임계값(confidence threshold)을 낮춰보세요.")
        
        # 요약 정보를 텍스트 파일로 저장
        summary_path = os.path.join(output_folder_path, f"{current_datetime}_{clean_model_name}_{clean_video_name}_sign_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_text))
        
        print(f"✅ 빈 결과 파일 저장 완료")
        print(f"   CSV 파일: {signs_csv_path}")
        print(f"   요약 파일: {summary_path}")
        
        # 표지판이 검지되지 않아도 결과 폴더 열기
        try:
            if os.name == 'nt':  # Windows
                os.startfile(output_folder_path)
            elif os.name == 'posix':  # macOS/Linux
                subprocess.run(['open', output_folder_path], check=True)
            print(f"📂 결과 폴더를 열었습니다: {output_folder_path}")
        except Exception as e:
            print(f"⚠️ 폴더 열기 실패: {e}")
        
        messagebox.showinfo("분석 완료", 
                           f"주행모드 분석이 완료되었습니다.\n\n"
                           f"결과 저장 위치: {output_folder_path}\n"
                           f"- 표지판 검지 CSV: 0개 기록 (빈 파일)\n"
                           f"- 분석 영상\n"
                           f"- 검지 요약 파일\n\n"
                           f"※ 표지판이 검지되지 않았습니다.\n"
                           f"모델이 표지판 검지용으로 훈련되었는지 확인해주세요.\n\n"
                           f"📂 결과 폴더가 자동으로 열렸습니다!")
        return

    # --- 데이터프레임 생성 ---
    df_crossing_data = pd.DataFrame(crossing_data)
    df_speed = pd.DataFrame(speed_records)
    df_all_vehicles = pd.DataFrame(presence_data)
    df_distance_deceleration = pd.DataFrame(distance_deceleration_records)

    if df_crossing_data.empty:
        df_crossing_data = pd.DataFrame(columns=['ID','Class','Time','Line','IsInitialLine'])

    avg_speed_df_cols = ['ID','Class','IsInitialLine','IsSteppedTrackingLine','Avg Speed(km/h)']
    if not df_speed.empty:
        avg_speed_df_temp = df_speed.groupby(['ID','Class','IsInitialLine','IsSteppedTrackingLine'], as_index=False).mean()
        if not avg_speed_df_temp.empty and 'Speed(km/h)' in avg_speed_df_temp.columns :
            avg_speed_df = avg_speed_df_temp[['ID','Class','IsInitialLine','IsSteppedTrackingLine', 'Speed(km/h)']].copy()
            avg_speed_df.rename(columns={'Speed(km/h)': 'Avg Speed(km/h)'}, inplace=True)
        else:
            avg_speed_df = pd.DataFrame(columns=avg_speed_df_cols)
    else:
        avg_speed_df = pd.DataFrame(columns=avg_speed_df_cols)

    df_presence = pd.DataFrame(presence_data) # df_all_vehicles와 동일할 수 있음, 필요시 정리

    if not df_crossing_data.empty:
        crossed_initial_ids = df_crossing_data[df_crossing_data['IsInitialLine'] == True]['ID'].unique()
        df_crossing_data.loc[df_crossing_data['ID'].isin(crossed_initial_ids), 'IsInitialLine'] = True
    else:
        crossed_initial_ids = np.array([])

    for df_to_update in [df_speed, df_all_vehicles, df_presence]: # df_presence 추가
        if not df_to_update.empty and 'ID' in df_to_update.columns and 'IsInitialLine' in df_to_update.columns:
            df_to_update.loc[df_to_update['ID'].isin(crossed_initial_ids), 'IsInitialLine'] = True
    
    # --- 새로운 상세 타임라인 로그 생성 ---
    # 1. 먼저 ID별로 최종적으로 밟은 tracking line을 수집
    vehicle_crossed_lines = defaultdict(set)
    if not df_crossing_data.empty:
        # 모든 tracking line 교차 이벤트에서 ID별로 통과한 라인 수집
        for _, row in df_crossing_data.iterrows():
            vehicle_id = row['ID']
            line_name = row['Line']
            if 'Tracking Line' in line_name:  # 'Tracking Line'이 포함된 이름만 필터링
                vehicle_crossed_lines[vehicle_id].add(line_name)
    
    # 2. 데이터 생성 - 각 ID에 대해 모든 시점의 로그를 생성하되, 최종적으로 tracking line을 밟은 차량만 포함
    timeline_logs = []
    
    # 3. 교차 이벤트 데이터를 시간별로 정리 (ID별 정확한 교차 시점 캡처를 위해)
    crossing_events_by_id_time = defaultdict(list)
    if not df_crossing_data.empty:
        for _, event in df_crossing_data.iterrows():
            vehicle_id = event['ID']
            if 'Tracking Line' in event['Line']:  # Tracking Line 이벤트만 저장
                crossing_events_by_id_time[(vehicle_id, event['Time'])].append(event['Line'])
    
    # 이미 ID별로 밟은 라인을 수집했으므로, 이제 해당 ID의 모든 기록 수집
    if not df_speed.empty and vehicle_crossed_lines:
        # tracking line을 밟은 모든 차량 ID 목록
        crossed_vehicles = set(vehicle_crossed_lines.keys())
        
        # speed_records 데이터에서 필요한 차량만 필터링
        for _, record in df_speed.iterrows():
            vehicle_id = record['ID']
            
            # 최종적으로 어떤 tracking line이라도 밟은 차량만 선택
            if vehicle_id in crossed_vehicles:
                time_point = record['Time']
                
                # 이 시점에 차량이 밟고 있는 라인 찾기
                current_line = "None"
                
                # 정확한 시간에 교차 이벤트가 있는지 확인 (정확한 교차 시점 캡처)
                if (vehicle_id, time_point) in crossing_events_by_id_time:
                    # 동일한 시간에 여러 라인을 밟았을 수도 있으므로 모두 포함
                    current_line = "; ".join(crossing_events_by_id_time[(vehicle_id, time_point)])
                
                # 최종적으로 이 차량이 밟은 모든 tracking line 목록 (정렬된 문자열)
                final_lines_crossed = "; ".join(sorted(vehicle_crossed_lines[vehicle_id]))
                
                timeline_logs.append({
                    'ID': vehicle_id,
                    'Time(s)': time_point,
                    'Class': record['Class'],
                    'Current_Line_Status': current_line,
                    'Speed(km/h)': record.get('Speed(km/h)', None),
                    'Acceleration(m/s²)': record.get('Acceleration(m/s²)', None),
                    'Deceleration(m/s²)': record.get('Deceleration(m/s²)', None),
                    'Final_Crossed_Lines': final_lines_crossed
                })
    
    # 데이터프레임으로 변환
    df_timeline = pd.DataFrame(timeline_logs)

    df_region = pd.DataFrame(region_records)
    if df_region.empty:
        df_region = pd.DataFrame(columns=['Region', 'ID', 'Class', 'SteppedTrackingLine', 'Speed(km/h)', 'Acceleration(m/s²)', 'Deceleration(m/s²)', 'LinesStepped'])

    df_region_line_stats_cols = ['Region', 'Class', 'SteppedTrackingLine_True', 'SteppedTrackingLine_False', 'TotalInRegion']
    df_region_line_stats = pd.DataFrame(columns=df_region_line_stats_cols)
    if not df_region.empty and all(col in df_region.columns for col in ['Region', 'ID', 'Class', 'SteppedTrackingLine']):
        df_region_unique = (
            df_region
            .groupby(['Region', 'ID', 'Class'], as_index=False)
            .agg(SteppedTrackingLine=('SteppedTrackingLine', 'max'))
        )
        summary_rows = []
        if not df_region_unique.empty:
            for (reg, cls), grp in df_region_unique.groupby(['Region', 'Class']):
                total = len(grp)
                stepped = grp['SteppedTrackingLine'].sum()
                not_stepped = total - stepped
                summary_rows.append({
                    'Region': reg, 'Class': cls,
                    'SteppedTrackingLine_True': int(stepped),
                    'SteppedTrackingLine_False': int(not_stepped),
                    'TotalInRegion': int(total)
                })
        if summary_rows:
             df_region_line_stats = pd.DataFrame(summary_rows, columns=df_region_line_stats_cols)


    df_region_summary = pd.DataFrame(
        [{"Region": idx + 1, "TotalUniqueVehiclesInRegion": len(ids)}
            for idx, ids in region_unique_ids.items()]
    )
    if df_region_summary.empty:
        df_region_summary = pd.DataFrame(columns=["Region", "TotalUniqueVehiclesInRegion"])

    hourly_data_cols = ['Hour', 'Cat1_CrossInit_NoStep', 'Cat2_CrossInit_Step', 'Cat3_NoCrossInit_NoStep', 'Cat4_NoCrossInit_Step']
    hourly_data = pd.DataFrame(columns=hourly_data_cols)
    category_dfs_names = ['category1_ids', 'category2_ids', 'category3_ids', 'category4_ids']
    # 각 카테고리별 ID 목록을 저장할 DataFrame 초기화 (CSV로 저장될 것임)
    category_id_dfs = {name: pd.DataFrame(columns=['Hour', 'Vehicle_ID', 'Class']) for name in category_dfs_names}

    # interval_df 초기화 (조건문 밖에서)
    interval_df_cols = ["Interval_Start", "Interval_End", "ID", "Class", "IsInitialLine", "IsSteppedTrackingLine"]
    interval_df = pd.DataFrame(columns=interval_df_cols)

    if not df_all_vehicles.empty and 'Time' in df_all_vehicles.columns and 'ID' in df_all_vehicles.columns:
        df_all_vehicles['Hour'] = (df_all_vehicles['Time'] // 3600).astype(int)
        grouped_hourly_id_status = df_all_vehicles.groupby(['Hour','ID'], as_index=False).agg(
            IsInitialLine=('IsInitialLine', 'max'),
            IsSteppedTrackingLine=('IsSteppedTrackingLine', 'max'),
            Class=('Class', 'first')
        )

        if not grouped_hourly_id_status.empty:
            category_id_dfs['category1_ids'] = grouped_hourly_id_status[
                (grouped_hourly_id_status['IsInitialLine'] == True) &
                (grouped_hourly_id_status['IsSteppedTrackingLine'] == False)
            ][['Hour','ID', 'Class']].drop_duplicates().rename(columns={'ID': 'Vehicle_ID'})

            category_id_dfs['category2_ids'] = grouped_hourly_id_status[
                (grouped_hourly_id_status['IsInitialLine'] == True) &
                (grouped_hourly_id_status['IsSteppedTrackingLine'] == True)
            ][['Hour','ID', 'Class']].drop_duplicates().rename(columns={'ID': 'Vehicle_ID'})

            category_id_dfs['category3_ids'] = grouped_hourly_id_status[
                (grouped_hourly_id_status['IsInitialLine'] == False) &
                (grouped_hourly_id_status['IsSteppedTrackingLine'] == False)
            ][['Hour','ID', 'Class']].drop_duplicates().rename(columns={'ID': 'Vehicle_ID'})

            category_id_dfs['category4_ids'] = grouped_hourly_id_status[
                (grouped_hourly_id_status['IsInitialLine'] == False) &
                (grouped_hourly_id_status['IsSteppedTrackingLine'] == True)
            ][['Hour','ID', 'Class']].drop_duplicates().rename(columns={'ID': 'Vehicle_ID'})

            hourly_summary_counts_dict = {
                'Cat1_CrossInit_NoStep': grouped_hourly_id_status[(grouped_hourly_id_status['IsInitialLine'] == True) & (grouped_hourly_id_status['IsSteppedTrackingLine'] == False)].groupby('Hour')['ID'].nunique(),
                'Cat2_CrossInit_Step': grouped_hourly_id_status[(grouped_hourly_id_status['IsInitialLine'] == True) & (grouped_hourly_id_status['IsSteppedTrackingLine'] == True)].groupby('Hour')['ID'].nunique(),
                'Cat3_NoCrossInit_NoStep': grouped_hourly_id_status[(grouped_hourly_id_status['IsInitialLine'] == False) & (grouped_hourly_id_status['IsSteppedTrackingLine'] == False)].groupby('Hour')['ID'].nunique(),
                'Cat4_NoCrossInit_Step': grouped_hourly_id_status[(grouped_hourly_id_status['IsInitialLine'] == False) & (grouped_hourly_id_status['IsSteppedTrackingLine'] == True)].groupby('Hour')['ID'].nunique()
            }
            hourly_data_temp = pd.DataFrame(hourly_summary_counts_dict).reset_index()
            hourly_data_temp.fillna(0, inplace=True)
            for col in hourly_data_cols:
                if col != 'Hour' and col in hourly_data_temp.columns : hourly_data_temp[col] = hourly_data_temp[col].astype(int)
            hourly_data = hourly_data_temp.reindex(columns=hourly_data_cols).fillna(0)

        # df_presence는 df_all_vehicles와 내용이 중복될 수 있으므로 df_all_vehicles를 사용
        if not df_all_vehicles.empty and 'Time' in df_all_vehicles.columns:
            max_time_val = df_all_vehicles["Time"].max()
            intervals = np.arange(0, max_time_val + 5, 5)
            interval_records = []
            if len(intervals) > 1:
                for start_t in intervals[:-1]:
                    end_t = start_t + 5
                    # df_presence 대신 df_all_vehicles 사용
                    subset = df_all_vehicles[(df_all_vehicles["Time"] >= start_t) & (df_all_vehicles["Time"] < end_t)]
                    if not subset.empty and all(c in subset.columns for c in ['ID', 'Class', 'IsInitialLine', 'IsSteppedTrackingLine']):
                        grouped_subset_interval = subset.groupby(['ID','Class'], as_index=False).agg(
                            IsInitialLine=('IsInitialLine', 'max'),
                            IsSteppedTrackingLine=('IsSteppedTrackingLine', 'max')
                        )
                        for _, row in grouped_subset_interval.iterrows():
                            interval_records.append({
                                "Interval_Start": start_t, "Interval_End": end_t,
                                "ID": row["ID"], "Class": row["Class"],
                                "IsInitialLine": row["IsInitialLine"],
                                "IsSteppedTrackingLine": row["IsSteppedTrackingLine"]
                            })
            if interval_records:
                 interval_df = pd.DataFrame(interval_records, columns=interval_df_cols)
            else:
                 interval_df = pd.DataFrame(columns=interval_df_cols)


    total_duration_hours = (total_frames / fps / 3600) if fps > 0 else 0

    total_vehicles_all_unique = df_all_vehicles['ID'].nunique() if not df_all_vehicles.empty and 'ID' in df_all_vehicles.columns else 0
    total_vehicles_cross_O_unique = df_all_vehicles[(df_all_vehicles['IsInitialLine'] == True) & ('ID' in df_all_vehicles.columns)]['ID'].nunique() if not df_all_vehicles.empty else 0
    total_vehicles_not_cross_O_unique = total_vehicles_all_unique - total_vehicles_cross_O_unique

    traffic_flow_all_ph = (total_vehicles_all_unique / total_duration_hours) if total_duration_hours > 0 else 0
    traffic_flow_cross_O_ph = (total_vehicles_cross_O_unique / total_duration_hours) if total_duration_hours > 0 else 0
    traffic_flow_not_cross_O_ph = (total_vehicles_not_cross_O_unique / total_duration_hours) if total_duration_hours > 0 else 0

    traffic_summary_overall = pd.DataFrame({
        'Metric': ['Total Unique Vehicles (Initial Line Crossed)', 'Total Unique Vehicles (Initial Line Not Crossed)',
                   'Total Unique Vehicles (All)', 'Traffic Flow (Initial Line Crossed) [vehicles/hour]',
                   'Traffic Flow (Initial Line Not Crossed) [vehicles/hour]', 'Traffic Flow (All Vehicles) [vehicles/hour]'],
        'Value': [total_vehicles_cross_O_unique, total_vehicles_not_cross_O_unique, total_vehicles_all_unique,
                  traffic_flow_cross_O_ph, traffic_flow_not_cross_O_ph, traffic_flow_all_ph]
    })

    traffic_summary_per_class_cols = ['Class', 'Vehicles_Crossed_Initial_Line', 'Vehicles_Not_Crossed_Initial_Line', 'Total_Vehicles',
                                      'Traffic_Flow_All_Vehicles_[vehicles/hour]', 'Traffic_Flow_Crossed_Initial_Line_[vehicles/hour]',
                                      'Traffic_Flow_Not_Crossed_Initial_Line_[vehicles/hour]']
    traffic_summary_per_class = pd.DataFrame(columns=traffic_summary_per_class_cols)
    if not df_all_vehicles.empty and total_duration_hours > 0 and all(c in df_all_vehicles.columns for c in ['Class', 'ID', 'IsInitialLine']):
        class_summary_status = df_all_vehicles.groupby(['Class', 'ID'], as_index=False).agg(
            IsInitialLine=('IsInitialLine', 'max')
        ).groupby('Class', as_index=False).agg(
            Total_Vehicles=('ID', 'nunique'),
            Vehicles_Crossed_Initial_Line=('IsInitialLine', lambda x: x.astype(bool).sum())
        )
        if not class_summary_status.empty:
            class_summary_status['Vehicles_Not_Crossed_Initial_Line'] = class_summary_status['Total_Vehicles'] - class_summary_status['Vehicles_Crossed_Initial_Line']
            class_summary_status['Traffic_Flow_All_Vehicles_[vehicles/hour]'] = class_summary_status['Total_Vehicles'] / total_duration_hours
            class_summary_status['Traffic_Flow_Crossed_Initial_Line_[vehicles/hour]'] = class_summary_status['Vehicles_Crossed_Initial_Line'] / total_duration_hours
            class_summary_status['Traffic_Flow_Not_Crossed_Initial_Line_[vehicles/hour]'] = class_summary_status['Vehicles_Not_Crossed_Initial_Line'] / total_duration_hours
            traffic_summary_per_class = class_summary_status.reindex(columns=traffic_summary_per_class_cols, fill_value=0)


    df_cross_for_profile = df_crossing_data.copy()
    profile_cols = ['ID', 'Class', 'First_Cross', 'Last_Cross', 'Lines_List', 'Lines_Count', 'Avg_Speed', 'Peak_Speed']
    profile = pd.DataFrame(columns=profile_cols)
    if not df_cross_for_profile.empty and all(c in df_cross_for_profile.columns for c in ['ID', 'Class', 'Time', 'Line']):
        profile_temp = (
            df_cross_for_profile.groupby('ID', as_index=False).agg(
                Class=('Class', 'first'), First_Cross=('Time', 'min'), Last_Cross=('Time', 'max'),
                Lines_List=('Line', lambda x: ';'.join(sorted(x.unique()))), Lines_Count=('Line', 'nunique')))
        if not df_speed.empty and 'ID' in df_speed.columns and 'Speed(km/h)' in df_speed.columns:
            spd_stat = (df_speed.groupby('ID', as_index=False)['Speed(km/h)'].agg(Avg_Speed='mean', Peak_Speed='max'))
            profile_temp = profile_temp.merge(spd_stat, on='ID', how='left')
        else:
            profile_temp['Avg_Speed'] = np.nan
            profile_temp['Peak_Speed'] = np.nan
        profile = profile_temp.reindex(columns=profile_cols)


    pivot_cols = ['Lines_Count'] # 실제 피벗 결과에 따라 컬럼이 동적으로 추가됨
    pivot = pd.DataFrame(columns=pivot_cols) # 초기화
    if not profile.empty and all(c in profile.columns for c in ['Lines_Count', 'Class', 'ID']):
        try:
            pivot_temp = (profile.pivot_table(index='Lines_Count', columns='Class', values='ID', aggfunc='nunique', fill_value=0))
            pivot = pivot_temp.reset_index() # 'Lines_Count'를 컬럼으로 만듦
            # 피벗 테이블의 컬럼은 ['Lines_Count', 'car', 'truck', ...] 등이 됨
        except Exception as e:
            print(f"피벗 테이블 생성 오류: {e}")


    bucket_cols = ['Bucket', 'Vehicle_Count', 'Avg_Speed']
    bucket = pd.DataFrame(columns=bucket_cols)
    if not profile.empty and all(c in profile.columns for c in ['Lines_Count', 'ID']):
        profile_b = profile.copy()
        profile_b['Bucket'] = np.where(profile_b['Lines_Count'] >= 3, '3+', profile_b['Lines_Count'].astype(str))
        agg_config_bucket = {'Vehicle_Count': ('ID', 'nunique')}
        if 'Avg_Speed' in profile_b.columns:
            agg_config_bucket['Avg_Speed'] = ('Avg_Speed', 'mean')
        bucket_temp = profile_b.groupby('Bucket', as_index=False).agg(**agg_config_bucket)
        bucket = bucket_temp.reindex(columns=bucket_cols)


    tracking_line_by_class_df_cols = ['Line', 'Class', 'Count']
    tracking_line_by_class_df = pd.DataFrame(columns=tracking_line_by_class_df_cols)
    if not df_crossing_data.empty and all(c in df_crossing_data.columns for c in ['Line', 'Class']):
        try:
            # ID별로 한 번만 카운트하려면 nunique() 사용
            # grouped_size_series = df_crossing_data.groupby(['Line', 'Class'])['ID'].nunique()
            # 만약 각 통과 이벤트를 모두 카운트하려면 size() 사용
            grouped_size_series = df_crossing_data.groupby(['Line', 'Class']).size()
            df_with_count = grouped_size_series.reset_index(name='Count')
            tracking_line_by_class_df = df_with_count.reindex(columns=tracking_line_by_class_df_cols)
        except Exception as e_groupby_size:
            print(f"Error creating 'tracking_line_by_class_df': {e_groupby_size}")
            tracking_line_by_class_df = pd.DataFrame(columns=tracking_line_by_class_df_cols)


    df_id_avg_cols = ['ID', 'Class', 'IsInitialLine', 'Speed(km/h)', 'Acceleration(m/s²)',
                      'Deceleration(m/s²)', 'Acceleration (%)', 'Deceleration (%)']
    df_id_avg = pd.DataFrame(columns=df_id_avg_cols)
    if not df_distance_deceleration.empty:
        df_id1_list, df_id2_list = [], []
        map_id1 = {'ID1':'ID', 'Class1':'Class', 'Speed1(km/h)':'Speed(km/h)', 'Acceleration1(m/s²)':'Acceleration(m/s²)',
                   'Deceleration1(m/s²)':'Deceleration(m/s²)', 'Acceleration1 (%)':'Acceleration (%)',
                   'Deceleration1 (%)':'Deceleration (%)', 'IsInitialLine_ID1':'IsInitialLine'}
        map_id2 = {'ID2':'ID', 'Class2':'Class', 'Speed2(km/h)':'Speed(km/h)', 'Acceleration2(m/s²)':'Acceleration(m/s²)',
                   'Deceleration2(m/s²)':'Deceleration(m/s²)', 'Acceleration2 (%)':'Acceleration (%)',
                   'Deceleration2 (%)':'Deceleration (%)', 'IsInitialLine_ID2':'IsInitialLine'}

        for _, row_dd in df_distance_deceleration.iterrows():
            id1_data = {target_col: row_dd.get(source_col, np.nan) for source_col, target_col in map_id1.items()}
            df_id1_list.append(id1_data)
            id2_data = {target_col: row_dd.get(source_col, np.nan) for source_col, target_col in map_id2.items()}
            df_id2_list.append(id2_data)

        df_all_ids_for_avg = pd.concat([pd.DataFrame(df_id1_list), pd.DataFrame(df_id2_list)], ignore_index=True)

        if not df_all_ids_for_avg.empty and all(c in df_all_ids_for_avg.columns for c in ['ID', 'Class', 'IsInitialLine']):
            group_keys = ['ID', 'Class', 'IsInitialLine']
            numeric_cols_for_mean = [col for col in df_id_avg_cols if col not in group_keys and col in df_all_ids_for_avg.columns]

            if numeric_cols_for_mean:
                df_id_avg_temp = df_all_ids_for_avg.groupby(group_keys, as_index=False)[numeric_cols_for_mean].mean()
                df_id_avg = df_id_avg_temp.reindex(columns=df_id_avg_cols)
            else:
                df_id_avg_temp = df_all_ids_for_avg[group_keys].drop_duplicates().reset_index(drop=True)
                df_id_avg = df_id_avg_temp.reindex(columns=df_id_avg_cols, fill_value=np.nan)

    # --- CSV 파일 저장 로직 ---
    def save_df_to_csv(df, file_name_suffix):
        if not df.empty:
            # 파일명을 더 짧게 구성 - 날짜는 간소화, 비디오명은 축약
            date_short = current_datetime.split('_')[0]  # 날짜만 추출 (시간 제외)
            safe_file_name_suffix = sanitize_filename(file_name_suffix, 30)
            safe_video_name_csv = sanitize_filename(video_name_without_ext, 40)
            safe_model_name_csv = sanitize_filename(model_name.replace('.pt', ''), 15)
            
            file_name = f"{date_short}_{safe_model_name_csv}_{safe_video_name_csv}_{safe_file_name_suffix}.csv"
            file_name = sanitize_filename(file_name, 150)  # Windows 파일명 길이 제한 강화
            
            file_path = os.path.join(output_folder_path, file_name)
            # 경로 정규화
            file_path = os.path.normpath(file_path)
            
            print(f"CSV 저장 시도: {file_path}")
            
            try:
                # 출력 폴더가 존재하는지 확인하고 없으면 생성
                if not os.path.exists(output_folder_path):
                    print(f"출력 폴더가 존재하지 않음. 생성 시도: {output_folder_path}")
                    os.makedirs(output_folder_path, exist_ok=True)
                    print(f"출력 폴더 생성 완료: {output_folder_path}")
                
                # 파일 디렉토리 확인
                file_dir = os.path.dirname(file_path)
                if not os.path.exists(file_dir):
                    print(f"파일 디렉토리 생성: {file_dir}")
                    os.makedirs(file_dir, exist_ok=True)
                
                # CSV 파일 저장
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                print(f"데이터가 성공적으로 저장됨: {file_path}")
                
                # 파일 저장 확인
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"저장된 파일 크기: {file_size} bytes")
                else:
                    print("경고: 파일이 저장되지 않았습니다.")
                    
            except PermissionError as e:
                print(f"권한 오류로 CSV 저장 실패: {e}")
                # 현재 디렉토리에 저장 시도
                fallback_path = os.path.join(os.getcwd(), file_name)
                try:
                    df.to_csv(fallback_path, index=False, encoding='utf-8-sig')
                    print(f"Fallback으로 현재 디렉토리에 저장됨: {fallback_path}")
                except Exception as e2:
                    print(f"Fallback 저장도 실패: {e2}")
            except Exception as e:
                print(f"{file_name_suffix} CSV 파일 저장 중 오류 발생: {e}")
                print(f"시도한 파일 경로: {file_path}")
                print(f"출력 폴더 존재 여부: {os.path.exists(output_folder_path)}")
                print(f"출력 폴더 쓰기 권한: {os.access(output_folder_path, os.W_OK) if os.path.exists(output_folder_path) else 'N/A'}")
                import traceback
                traceback.print_exc()
        else:
            print(f"{file_name_suffix} DataFrame이 비어 있어 CSV 파일을 생성하지 않습니다.")

    try:
        # 메타데이터 저장
        metadata_df = pd.DataFrame({
            "Info": ["Date/Time", "Video Name", "Model Name", "Analysis Duration (hours)", "FPS"],
            "Value": [current_datetime, video_name, model_name, f"{total_duration_hours:.2f}", f"{fps:.2f}"]
        })
        save_df_to_csv(metadata_df, "Metadata")

        save_df_to_csv(traffic_summary_overall, "OverallSummary")
        save_df_to_csv(traffic_summary_per_class, "ClassSummary")
        save_df_to_csv(df_crossing_data, "CrossingEvents")
        save_df_to_csv(df_speed, "SpeedLog")
        save_df_to_csv(avg_speed_df, "AvgSpeedPerID")
        save_df_to_csv(df_all_vehicles, "AllDetectionsLog") # df_presence와 내용이 유사하므로 하나만 저장
        save_df_to_csv(df_distance_deceleration, "InterVehicleDistance")
        save_df_to_csv(hourly_data, "HourlyCounts")
        save_df_to_csv(interval_df, "5secIntervalLog")

        for cat_idx, (cat_name, cat_df) in enumerate(category_id_dfs.items()):
            save_df_to_csv(cat_df, f"IDs_Cat{cat_idx+1}")

        save_df_to_csv(df_region, "RegionLog")
        save_df_to_csv(df_region_summary, "RegionSummaryCounts")
        save_df_to_csv(df_region_line_stats, "RegionLineCrossingStats")

        df_class_initial_pass_counts_list = []
        if unique_ids_per_class:
            for cls_item, total_ids_set in unique_ids_per_class.items():
                passed_initial_count = class_counts_O.get(cls_item, 0)
                not_passed_initial_count = len(total_ids_set) - passed_initial_count
                df_class_initial_pass_counts_list.append({
                    'Class': cls_item, 'Passed_Initial_Line': passed_initial_count,
                    'Not_Passed_Initial_Line': not_passed_initial_count, 'Total_Unique_IDs': len(total_ids_set)})
            if df_class_initial_pass_counts_list:
                save_df_to_csv(pd.DataFrame(df_class_initial_pass_counts_list), "ClassInitialLinePassCounts")

        lane_events_df = pd.DataFrame(lane_events, columns=["Frame", "Time", "Lane", "Status", "Count", "Details"])
        save_df_to_csv(lane_events_df, "LaneEvents")

        lane_thresh_list = []
        if lane_thresholds or lane_polygons:
             for i in range(len(lane_polygons)):
                 lane_thresh_list.append({'Lane': f'Lane {i+1}', 'Threshold': lane_thresholds.get(i, "N/A")})
        if lane_thresh_list:
             save_df_to_csv(pd.DataFrame(lane_thresh_list), "LaneThresholds")
        elif not lane_thresh_list and (lane_thresholds or lane_polygons):
             save_df_to_csv(pd.DataFrame(columns=['Lane', 'Threshold']), "LaneThresholds_Empty")


        save_df_to_csv(profile, "Vehicle_Line_Profile")
        save_df_to_csv(pivot, "LineCount_Pivot") # pivot의 컬럼은 동적이므로, CSV로 저장하면 모든 컬럼이 잘 저장됨
        save_df_to_csv(bucket, "Lines_Count_Buckets")
        save_df_to_csv(tracking_line_by_class_df, "TrackingLineByClass")
        save_df_to_csv(df_id_avg, "DecelDist_Avg_per_ID")
        
        # 새로운 CSV 파일 저장 - 차량별 타임라인 로그
        save_df_to_csv(df_timeline, "Vehicle_Timeline_Detail")

        print(f"Processing 완료. 결과 CSV 파일들은 {output_folder_path}에 저장되었습니다.")

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"CSV 파일 저장 중 오류 발생: {e}\n{error_details}")
        messagebox.showerror("오류", f"CSV 파일 저장 중 오류 발생: {e}\n\n세부 정보:\n{error_details[:500]}...")
        return

    messagebox.showinfo("완료", "분석 완료! 결과는 CSV 파일들로 저장되었습니다.")
    open_folder(output_folder_path)


def perform_setup(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        messagebox.showerror("오류", f"영상 파일을 열 수 없습니다: {input_video_path}")
        return None, None, None
    ret, frame_setup = cap.read()
    if not ret:
        messagebox.showerror("오류", "첫 프레임 읽기 실패")
        cap.release()
        return None, None, None

    cv2.namedWindow('Setup ROI', cv2.WINDOW_NORMAL)
    # setup_win_w = frame_setup.shape[1] // 2 # 필요시 주석 해제
    # setup_win_h = frame_setup.shape[0] // 2 # 필요시 주석 해제
    # cv2.resizeWindow('Setup ROI', setup_win_w, setup_win_h) # 필요시 주석 해제


    rect_points_setup = []
    current_dragging_point_idx = None

    def mouse_roi_setup(event, x, y, flags, param):
        nonlocal rect_points_setup, current_dragging_point_idx

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(rect_points_setup) < 4:
                rect_points_setup.append((x, y))
            else:
                for i, p_roi in enumerate(rect_points_setup):
                    if dist(p_roi, (x, y)) < Constants.DRAG_DETECTION_RADIUS:
                        current_dragging_point_idx = i
                        break

        elif event == cv2.EVENT_MOUSEMOVE:
            if current_dragging_point_idx is not None and flags & cv2.EVENT_FLAG_LBUTTON:
                if 0 <= current_dragging_point_idx < len(rect_points_setup):
                    rect_points_setup[current_dragging_point_idx] = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            current_dragging_point_idx = None

    cv2.setMouseCallback('Setup ROI', mouse_roi_setup)

    help_text_roi = "Click 4 points for ROI (e.g., TL, TR, BR, BL).\nDrag to adjust. 'r':Reset, 's':Save, 'q':Quit"

    while True:
        draw_frame_setup = frame_setup.copy()

        y_text_start = 30
        for i, line_text in enumerate(help_text_roi.split('\n')): # 변수명 변경 line -> line_text
            cv2.putText(draw_frame_setup, line_text, (30, y_text_start + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,200,50), 1)


        for i, p_roi_draw in enumerate(rect_points_setup):
            color = (0,0,255)
            cv2.circle(draw_frame_setup, p_roi_draw, 7, color, -1)
            cv2.putText(draw_frame_setup, str(i+1), (int(p_roi_draw[0])+10, int(p_roi_draw[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if len(rect_points_setup) == 4:
            pts_arr = np.array(rect_points_setup, dtype=np.int32)
            cv2.polylines(draw_frame_setup, [pts_arr], True, (0,255,0), 2)

        cv2.imshow('Setup ROI', draw_frame_setup)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            rect_points_setup.clear()
            current_dragging_point_idx = None
        elif key == ord('s'):
            if len(rect_points_setup) == 4:
                popup_root = tk.Tk()
                popup_root.withdraw()
                try:
                    width_m_input = simpledialog.askfloat("Input Real Width", "Enter REAL WORLD width of the ROI (meters):", parent=popup_root, minvalue=0.01)
                    if width_m_input is None: raise ValueError("Width input cancelled.")

                    height_m_input = simpledialog.askfloat("Input Real Height", "Enter REAL WORLD height of the ROI (meters):", parent=popup_root, minvalue=0.01)
                    if height_m_input is None: raise ValueError("Height input cancelled.")

                    popup_root.destroy()
                    cv2.destroyWindow('Setup ROI')
                    cap.release()
                    return rect_points_setup, width_m_input, height_m_input

                except ValueError as e:
                    messagebox.showerror("Error", f"Invalid input: {e}", parent=popup_root)
                    if popup_root.winfo_exists(): popup_root.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"An unexpected error occurred: {e}", parent=popup_root)
                    if popup_root.winfo_exists(): popup_root.destroy()
            else:
                messagebox.showwarning("Setup ROI", "Please define 4 points for the ROI first.")

        elif key == ord('q'):
            cv2.destroyWindow('Setup ROI')
            cap.release()
            return None, None, None

def perform_aerial_setup(input_video_path):
    """항공뷰 모드에서 다중 기준선 설정을 위한 함수"""
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        messagebox.showerror("오류", f"영상 파일을 열 수 없습니다: {input_video_path}")
        return None
    
    ret, frame_setup = cap.read()
    if not ret:
        messagebox.showerror("오류", "첫 프레임 읽기 실패")
        cap.release()
        return None
    
    cv2.namedWindow('Aerial View Setup', cv2.WINDOW_NORMAL)
    
    # 항공뷰 설정용 변수들
    scale_lines = []  # 여러 기준선을 저장
    current_line_points = []
    drawing_line = False
    start_point = None
    
    # 확대/축소 관련 변수
    zoom_factor = 1.0
    pan_x, pan_y = 0, 0
    original_frame = frame_setup.copy()
    
    def get_frame_with_zoom():
        """확대/축소가 적용된 프레임 반환"""
        if zoom_factor == 1.0:
            return original_frame.copy()
        
        # 확대된 프레임 생성
        h, w = original_frame.shape[:2]
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
        zoomed = cv2.resize(original_frame, (new_w, new_h))
        
        # 팬닝 적용
        start_x = max(0, min(new_w - w, pan_x))
        start_y = max(0, min(new_h - h, pan_y))
        end_x = min(new_w, start_x + w)
        end_y = min(new_h, start_y + h)
        
        return zoomed[start_y:end_y, start_x:end_x]
    
    def screen_to_original_coords(x, y):
        """화면 좌표를 원본 이미지 좌표로 변환"""
        if zoom_factor == 1.0:
            return x, y
        
        orig_x = int((x + pan_x) / zoom_factor)
        orig_y = int((y + pan_y) / zoom_factor)
        return orig_x, orig_y
    
    # 휠 드래그 팬닝 관련 변수
    panning = False
    last_pan_pos = None
    
    def aerial_setup_mouse_callback(event, x, y, flags, param):
        nonlocal current_line_points, drawing_line, start_point
        nonlocal panning, last_pan_pos, pan_x, pan_y
        
        # 화면 좌표를 원본 좌표로 변환
        orig_x, orig_y = screen_to_original_coords(x, y)
        
        if event == cv2.EVENT_MBUTTONDOWN:  # 휠(가운데) 버튼 눌림
            panning = True
            last_pan_pos = (x, y)
            print("팬닝 모드 시작 (휠 드래그)")
        
        elif event == cv2.EVENT_MBUTTONUP:  # 휠 버튼 뗌
            panning = False
            last_pan_pos = None
            print("팬닝 모드 종료")
        
        elif event == cv2.EVENT_MOUSEMOVE and panning:  # 휠 버튼 누른 채로 드래그
            if last_pan_pos is not None and zoom_factor > 1.0:
                # 마우스 이동량 계산
                dx = x - last_pan_pos[0]
                dy = y - last_pan_pos[1]
                
                # 팬닝 업데이트 (이동 방향 반대로)
                pan_x = max(0, pan_x - dx)
                pan_y = max(0, pan_y - dy)
                
                last_pan_pos = (x, y)
        
        elif event == cv2.EVENT_LBUTTONDOWN and not panning:  # 좌클릭 (팬닝 중이 아닐 때만)
            start_point = (orig_x, orig_y)
            drawing_line = True
        
        elif event == cv2.EVENT_LBUTTONUP and not panning:  # 좌클릭 뗌
            if drawing_line and start_point:
                # 새 라인 완성
                end_point = (orig_x, orig_y)
                line_length = math.sqrt((end_point[0] - start_point[0])**2 + 
                                      (end_point[1] - start_point[1])**2)
                if line_length > 5:  # 최소 길이 체크
                    current_line_points = [start_point, end_point]
                drawing_line = False
                start_point = None
        
        elif event == cv2.EVENT_MOUSEWHEEL:  # 마우스 휠 스크롤
            if flags > 0:  # 휠 업 (확대)
                zoom_factor = min(5.0, zoom_factor * 1.1)
                print(f"휠 확대: {zoom_factor:.1f}x")
            else:  # 휠 다운 (축소)
                zoom_factor = max(0.5, zoom_factor / 1.1)
                if zoom_factor == 1.0:
                    pan_x = pan_y = 0  # 1x일 때 팬 리셋
                print(f"휠 축소: {zoom_factor:.1f}x")
    
    cv2.setMouseCallback('Aerial View Setup', aerial_setup_mouse_callback)
    
    messagebox.showinfo("항공뷰 다중 기준선 설정", 
                        "항공뷰 다중 기준선 설정 (오차 감소)\n\n"
                        "🖱️ 마우스 조작:\n"
                        "• 좌클릭 드래그: 기준선 그리기 (차선 폭 3.5m 권장)\n"
                        "• 휠 스크롤: 확대/축소\n"
                        "• 휠 버튼 드래그: 화면 이동 (AutoCAD 방식)\n\n"
                        "⌨️ 키보드 조작:\n"
                        "• 'a' 키: 현재 선을 기준선 목록에 추가\n"
                        "• 'd' 키: 마지막 기준선 삭제\n"
                        "• '+/-' 키: 키보드로 확대/축소\n"
                        "• 'r' 키: 확대/팬 초기화\n"
                        "• Enter: 설정 완료 | q: 취소\n\n"
                        "💡 여러 개 기준선을 측정하면 평균값으로 정확도 향상!")
    
    while True:
        # 확대/축소가 적용된 프레임 가져오기
        frame_copy = get_frame_with_zoom()
        
        def original_to_screen_coords(orig_x, orig_y):
            """원본 좌표를 화면 좌표로 변환"""
            if zoom_factor == 1.0:
                return orig_x, orig_y
            screen_x = int(orig_x * zoom_factor - pan_x)
            screen_y = int(orig_y * zoom_factor - pan_y)
            return screen_x, screen_y
        
        # 저장된 기준선들 표시 (파란색)
        for i, line in enumerate(scale_lines):
            p1_screen = original_to_screen_coords(line['points'][0][0], line['points'][0][1])
            p2_screen = original_to_screen_coords(line['points'][1][0], line['points'][1][1])
            
            # 화면 범위 내에 있는 선만 그리기
            h, w = frame_copy.shape[:2]
            if (0 <= p1_screen[0] <= w and 0 <= p1_screen[1] <= h) or \
               (0 <= p2_screen[0] <= w and 0 <= p2_screen[1] <= h):
                cv2.line(frame_copy, p1_screen, p2_screen, (255, 0, 0), 3)  # 파란색
                cv2.circle(frame_copy, p1_screen, 5, (255, 0, 0), -1)
                cv2.circle(frame_copy, p2_screen, 5, (255, 0, 0), -1)
                
                # 기준선 번호 표시
                mid_point = ((p1_screen[0] + p2_screen[0]) // 2,
                           (p1_screen[1] + p2_screen[1]) // 2)
                cv2.putText(frame_copy, f"Line {i+1}: {line['length']:.1f}px", 
                          (mid_point[0] - 60, mid_point[1] - 15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 현재 그리고 있는 기준선 표시 (초록색)
        if len(current_line_points) == 2:
            p1_screen = original_to_screen_coords(current_line_points[0][0], current_line_points[0][1])
            p2_screen = original_to_screen_coords(current_line_points[1][0], current_line_points[1][1])
            
            cv2.line(frame_copy, p1_screen, p2_screen, (0, 255, 0), 3)  # 초록색
            cv2.circle(frame_copy, p1_screen, 5, (0, 255, 0), -1)
            cv2.circle(frame_copy, p2_screen, 5, (0, 255, 0), -1)
            
            # 현재 선 길이 표시
            line_length = math.sqrt((current_line_points[1][0] - current_line_points[0][0])**2 + 
                                  (current_line_points[1][1] - current_line_points[0][1])**2)
            mid_point = ((p1_screen[0] + p2_screen[0]) // 2,
                        (p1_screen[1] + p2_screen[1]) // 2)
            cv2.putText(frame_copy, f"Current: {line_length:.1f}px", 
                       (mid_point[0] - 60, mid_point[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 상태 정보 표시
        info_text = f"Lines: {len(scale_lines)} | Zoom: {zoom_factor:.1f}x"
        cv2.putText(frame_copy, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 도움말 표시
        help_text = "a:Add  d:Delete  Wheel:Zoom  MiddleDrag:Pan  r:Reset  Enter:Done"
        cv2.putText(frame_copy, help_text, (10, frame_copy.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 팬닝 모드 표시
        if panning:
            cv2.putText(frame_copy, "PANNING MODE", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 안내 텍스트
        cv2.putText(frame_copy, "Drag to draw reference line", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_copy, "Press Enter to confirm, q to cancel", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Aerial View Setup', frame_copy)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('a'):  # 현재 선을 기준선 목록에 추가
            if len(current_line_points) == 2:
                line_length = math.sqrt((current_line_points[1][0] - current_line_points[0][0])**2 + 
                                      (current_line_points[1][1] - current_line_points[0][1])**2)
                scale_lines.append({
                    'points': current_line_points.copy(),
                    'length': line_length
                })
                current_line_points.clear()
                print(f"기준선 추가됨. 총 {len(scale_lines)}개 기준선")
            else:
                print("먼저 기준선을 그려주세요.")
        
        elif key == ord('d'):  # 마지막 기준선 삭제
            if scale_lines:
                scale_lines.pop()
                print(f"마지막 기준선 삭제. 총 {len(scale_lines)}개 기준선")
            else:
                print("삭제할 기준선이 없습니다.")
        
        elif key == ord('+') or key == ord('='):  # 확대
            zoom_factor = min(5.0, zoom_factor * 1.2)
            print(f"확대: {zoom_factor:.1f}x")
        
        elif key == ord('-') or key == ord('_'):  # 축소
            zoom_factor = max(0.5, zoom_factor / 1.2)
            if zoom_factor == 1.0:
                pan_x = pan_y = 0  # 1x일 때 팬 리셋
            print(f"축소: {zoom_factor:.1f}x")
        
        elif key == ord('r'):  # 확대/팬 초기화
            zoom_factor = 1.0
            pan_x = pan_y = 0
            print("확대/팬 초기화")
        
        elif key == 13:  # Enter 키 - 설정 완료
            if len(scale_lines) > 0:
                # 거리 입력 받기
                distance_input = simpledialog.askfloat("스케일 설정", 
                                                     f"기준선 {len(scale_lines)}개의 실제 거리를 입력하세요:\n"
                                                     "(예: 차선 폭 3.5m)", 
                                                     minvalue=0.1, maxvalue=1000.0)
                if distance_input:
                    # 여러 기준선의 평균 픽셀 거리 계산
                    total_pixel_length = sum(line['length'] for line in scale_lines)
                    avg_pixel_length = total_pixel_length / len(scale_lines)
                    
                    meters_per_pixel = distance_input / avg_pixel_length
                    
                    # 정확도 정보 표시
                    pixel_lengths = [line['length'] for line in scale_lines]
                    std_dev = (sum((x - avg_pixel_length)**2 for x in pixel_lengths) / len(pixel_lengths))**0.5
                    accuracy_percent = (1 - std_dev / avg_pixel_length) * 100 if avg_pixel_length > 0 else 0
                    
                    cv2.destroyWindow('Aerial View Setup')
                    cap.release()
                    
                    messagebox.showinfo("스케일 설정 완료", 
                                      f"기준선 {len(scale_lines)}개 평균값 사용\n"
                                      f"평균 픽셀 길이: {avg_pixel_length:.1f}px\n"
                                      f"스케일: {meters_per_pixel:.6f} m/px\n"
                                      f"측정 정확도: {accuracy_percent:.1f}%")
                    
                    return meters_per_pixel
                else:
                    messagebox.showwarning("경고", "거리를 입력해주세요.")
            else:
                messagebox.showwarning("경고", "'a' 키로 기준선을 하나 이상 추가해주세요.")
        
        elif key == ord('q'):
            cv2.destroyWindow('Aerial View Setup')
            cap.release()
            return None

def lane_setup(frame_for_lane):
    global lane_polygons, lane_thresholds
    lane_polygons_temp = []
    lane_thresholds_temp = {}

    current_lane_points = []

    def lane_mouse_callback(event, x, y, flags, param):
        nonlocal current_lane_points
        if event == cv2.EVENT_LBUTTONDOWN:
            current_lane_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(current_lane_points) >= 3:
                lane_idx_temp = len(lane_polygons_temp)
                lane_polygons_temp.append(np.array(current_lane_points, dtype=np.int32))

                popup_root_lane = tk.Tk(); popup_root_lane.withdraw()
                threshold_val = simpledialog.askfloat("Lane Threshold", f"Enter threshold for Lane {lane_idx_temp + 1}:", parent=popup_root_lane, minvalue=0)
                popup_root_lane.destroy()

                if threshold_val is not None:
                    lane_thresholds_temp[lane_idx_temp] = threshold_val
                else:
                    lane_polygons_temp.pop()
                    messagebox.showinfo("Lane Setup", f"Lane {lane_idx_temp + 1} setup cancelled (no threshold).")
                current_lane_points.clear()
            else:
                messagebox.showwarning("Lane Setup", "A lane polygon needs at least 3 points.")

    cv2.namedWindow('Lane Setup', cv2.WINDOW_NORMAL)
    # setup_win_w_lane = frame_for_lane.shape[1] // 2 # 필요시 주석 해제
    # setup_win_h_lane = frame_for_lane.shape[0] // 2 # 필요시 주석 해제
    # cv2.resizeWindow('Lane Setup', setup_win_w_lane, setup_win_h_lane) # 필요시 주석 해제
    cv2.setMouseCallback('Lane Setup', lane_mouse_callback)

    help_text_lane = "Lanes: LClick=add pt, RClick=finish poly. 's':Save&Exit, 'r':Reset last, 'c':Clear all, 'q':Quit"

    while True:
        display_frame_lane = frame_for_lane.copy()
        y_text_start_lane = 20
        for i, line_text in enumerate(help_text_lane.split('\n')): # 변수명 변경 line -> line_text
             cv2.putText(display_frame_lane, line_text, (20, y_text_start_lane + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,200,50), 1)


        if len(current_lane_points) > 0:
            pts_current_lane = np.array(current_lane_points, dtype=np.int32)
            cv2.polylines(display_frame_lane, [pts_current_lane], False, (0,255,255), 2)
            for pt_cl in current_lane_points:
                cv2.circle(display_frame_lane, pt_cl, 4, (0,255,255), -1)

        for idx_lp, poly_pts_lp in enumerate(lane_polygons_temp):
            cv2.polylines(display_frame_lane, [poly_pts_lp], True, (255,0,0), 2)
            text_pos_lp = tuple(np.mean(poly_pts_lp, axis=0, dtype=np.int32))
            thresh_lp = lane_thresholds_temp.get(idx_lp, "N/A")
            cv2.putText(display_frame_lane, f"L{idx_lp+1}({thresh_lp})", text_pos_lp, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow('Lane Setup', display_frame_lane)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            lane_polygons = lane_polygons_temp
            lane_thresholds = lane_thresholds_temp
            messagebox.showinfo("Lane Setup", f"{len(lane_polygons)} lane(s) saved.")
            break
        elif key == ord('r'):
            if current_lane_points: current_lane_points.clear()
            elif lane_polygons_temp:
                removed_idx = len(lane_polygons_temp) - 1
                lane_polygons_temp.pop()
                if removed_idx in lane_thresholds_temp: del lane_thresholds_temp[removed_idx]
        elif key == ord('c'):
            current_lane_points.clear(); lane_polygons_temp.clear(); lane_thresholds_temp.clear()
        elif key == ord('q'):
            messagebox.showinfo("Lane Setup", "Lane setup quit. No changes saved to global variables.")
            break

    cv2.destroyWindow('Lane Setup')


def start_analysis(input_video_path, model_path, help_text_main_unused, analysis_mode="homography"):
    # 테스트 모드가 아닌 경우에만 영상 파일 체크
    if analysis_mode != "test":
        if not input_video_path or not model_path:
            messagebox.showwarning("경고", "영상 파일과 모델 파일을 모두 선택해주세요.")
            return
    else:
        # 테스트 모드에서는 모델 파일만 체크
        if not model_path:
            messagebox.showwarning("경고", "모델 파일을 선택해주세요.")
            return

    if analysis_mode == "test":
        # 테스트 모드용 출력 폴더 생성
        output_folder_path, current_datetime = create_test_output_folder(model_path)
    else:
        output_folder_path, current_datetime, video_name_str, model_name_str, video_name_no_ext_str = \
            create_output_folder(input_video_path, model_path)

    global aerial_view_mode, aerial_meters_per_pixel, driving_mode
    
    if analysis_mode == "test":
        # 테스트 모드: PT 모델 + 이미지 테스트
        if not model_path:
            messagebox.showwarning("경고", "테스트 모드에서는 PT 모델 파일이 필요합니다.")
            return
        
        # 테스트 이미지 선택
        test_images = filedialog.askopenfilenames(
            title="테스트할 이미지 파일들 선택",
            filetypes=[
                ("이미지 파일", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("모든 파일", "*.*")
            ]
        )
        
        if not test_images:
            messagebox.showinfo("정보", "테스트 이미지가 선택되지 않았습니다.")
            return
        
        messagebox.showinfo("테스트 모드", 
                           f"PT 모델 + 이미지 테스트 모드입니다.\n"
                           f"선택된 이미지: {len(test_images)}장\n"
                           f"사용할 모델: {os.path.basename(model_path)}")
        
        run_image_test_mode(test_images, model_path, output_folder_path, current_datetime)
        return
    elif analysis_mode == "driving":
        # 주행모드: 표지판 검지 전용
        driving_mode = True
        aerial_view_mode = False
        aerial_meters_per_pixel = None
        
        # 더미 ROI 데이터 (기존 함수 호환을 위해)
        rect_points_roi = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        width_m_val = 1.0   # 더미 값
        height_m_val = 1.0  # 더미 값
        
        messagebox.showinfo("주행모드 시작", 
                           "주행모드가 시작됩니다.\n"
                           "표지판 검지 시 자동으로 스크린샷을 캡처하고\n"
                           "CSV 파일로 데이터를 저장합니다.")
    elif analysis_mode == "aerial":
        # 항공뷰 모드: 기준선 설정 먼저 수행
        aerial_view_mode = True
        aerial_meters_per_pixel = perform_aerial_setup(input_video_path)
        if aerial_meters_per_pixel is None:
            messagebox.showinfo("정보", "항공뷰 설정이 취소되어 분석을 시작할 수 없습니다.")
            return
        
        # 더미 ROI 데이터 (호모그래피 함수 호출을 위해 필요)
        rect_points_roi = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        width_m_val = 1.0   # 더미 값 (사용되지 않음)
        height_m_val = 1.0  # 더미 값 (사용되지 않음)
        
        messagebox.showinfo("항공뷰 설정 완료", 
                           f"항공뷰 스케일이 설정되었습니다.\n"
                           f"스케일: {aerial_meters_per_pixel:.6f} m/pixel\n\n"
                           "이제 영상 분석을 시작합니다.")
    else:
        # 호모그래피 모드: 기존 방식
        driving_mode = False
        aerial_view_mode = False
        aerial_meters_per_pixel = None
        rect_points_roi, width_m_val, height_m_val = perform_setup(input_video_path)
        if rect_points_roi is None:
            messagebox.showinfo("정보", "ROI 설정이 취소되어 분석을 시작할 수 없습니다.")
            return

    cap_temp_lane = cv2.VideoCapture(input_video_path)
    ret_lane, first_frame_lane = cap_temp_lane.read()
    cap_temp_lane.release()

    global lane_polygons, lane_thresholds
    lane_polygons = []
    lane_thresholds = {}

    if analysis_mode == "aerial":
        # 항공뷰 모드에서는 차선 설정 스킵 (필요시 나중에 추가 가능)
        pass
    elif analysis_mode == "driving":
        # 주행모드에서는 차선 설정 스킵
        pass
    else:
        # 호모그래피 모드에서만 차선 설정
        if ret_lane:
            lane_setup(first_frame_lane)
        else:
            messagebox.showwarning("오류", "차선 설정을 위한 첫 프레임을 읽는 데 실패했습니다. 차선 설정 없이 진행됩니다.")

    global tracking_lines
    tracking_lines = []

    analysis_thread = threading.Thread(target=analysis_process, args=(
        input_video_path, model_path, output_folder_path, current_datetime,
        video_name_str, model_name_str, video_name_no_ext_str,
        "", # help_text는 analysis_process에서 사용 안 함
        rect_points_roi, width_m_val, height_m_val
    ))
    analysis_thread.start()
    messagebox.showinfo("분석 시작", "영상 분석이 백그라운드에서 시작되었습니다. 완료 시 알림이 표시됩니다.")


def main():
    root = tk.Tk()
    root.title("Traffic Super Vision (CSV Output Ver.)") # 제목에 CSV 명시
    root.geometry("700x320")  # 높이 증가

    video_path_var = tk.StringVar()
    model_path_var = tk.StringVar()
    analysis_mode_var = tk.StringVar(value="homography")  # 기본값: 호모그래피

    help_text_content = (
        '이 프로그램은 YOLO 모델을 사용하여 비디오에서 객체를 추적하고 분석합니다.\n'
        '드론(drone, uav, aircraft, helicopter) 검출 시 보라색으로 강조 표시됩니다.\n\n'
        '--- 설정 단계 ---\n'
        '1. [분석할 영상 파일 선택]: MP4, AVI 등의 영상 파일을 선택합니다.\n'
        '2. [사용할 모델 파일 선택]: YOLO .pt 모델 파일을 선택합니다.\n'
        '3. [분석 모드 선택]:\n'
        '   ★ 호모그래피 모드: 기존 방식 (일반 카메라, 고정 위치)\n'
        '     - ROI 4개 점 설정 → 실제 가로/세로 길이 입력\n'
        '   ★ 항공뷰 모드: 드론 촬영용 (수직 촬영, 균일한 스케일)\n'
        '     - 실행 후 기준선 하나만 그리고 실제 거리 입력\n'
        '   ★ 주행모드: 운전 중 촬영 영상 (표지판 검지 및 분석)\n'
        '     - 표지판 검지 시 자동 스크린샷 캡처\n'
        '     - 바운딩 박스 표시 및 CSV 데이터 저장\n'
        '   ★ 테스트 모드: PT 모델 성능 테스트 (영상 불필요)\n'
        '     - 이미지 파일들로 모델 검지 성능 확인\n'
        '     - 실시간 결과 표시 및 상세 분석 리포트\n'
        '4. 호모그래피 모드 선택 시:\n'
        '   - ROI 설정 창에서 네 점을 찍어 분석할 도로 영역 지정\n'
        '   - 실제 가로/세로 길이를 미터 단위로 입력\n'
        '   - 차선 설정 창에서 차선별 영역 설정 (선택사항)\n'
        '5. 항공뷰 모드 선택 시:\n'
        '   - 바로 분석 시작됨\n'
        '   - 실행 중에 마우스로 기준선 그리고 실제 거리 입력\n\n'
        '--- 분석 중 상호작용 (YOLO Detection 창) ---\n'
        '공통 기능:\n'
        '• 마우스 우클릭: 검지선(Tracking Line) 설정\n'
        '• 마우스 휠클릭: 초기 검지선(Initial Line) 설정\n'
        '• \'n\' 키: 사용자 정의 영역 그리기\n'
        '• \'p\' 키: 정보 표시 토글\n'
        '• \'t\' 키: 실시간 객체 이동 경로 표시 토글\n'
        '• \'i\' 키: 누적 trail 스택 표시 토글 (차로 사용량 분석)\n'
        '• \'q\' 키: 분석 중단\n'
        '• 영상 분석 완료 시 자동으로 누적 trail 이미지 저장\n\n'
        '항공뷰 모드 전용:\n'
        '• 마우스 좌클릭: 기준선 그리기 (시작점→끝점)\n'
        '• 거리 입력 창: 기준선의 실제 거리(m) 입력\n'
        '• \'r\' 키: 스케일 초기화\n\n'
        '호모그래피 모드 전용:\n'
        '• \'a\' 키: 항공뷰 모드로 전환 가능\n\n'
        '--- 결과물 ---\n'
        '분석 완료 후, 원본 영상이 있는 폴더 내에 결과 영상(.avi)과 다수의 분석 데이터 CSV 파일(.csv)이 저장된 새 폴더가 생성됩니다.\n'
        'CSV 파일들은 차량별 통행 기록, 속도, 가감속도, 차간 거리, 시간대별 통계 등 다양한 분석 결과를 포함합니다.\n\n'
        '--- 개발자 연락처 ---\n'
        '노우현 / 010-6886-0368 / KaKaoTalk ID: NoHyung\n'
        'GitHub: https://github.com/imonkfcwifi (imonkfcwifi)'
    )

    def browse_video():
        file_path = filedialog.askopenfilename(title="분석할 영상 파일 선택",
                                               filetypes=[("Video 파일", "*.mp4 *.avi *.mov *.mkv"), ("모든 파일", "*.*")])
        if file_path: video_path_var.set(file_path)

    def browse_model():
        file_path = filedialog.askopenfilename(title="사용할 모델 파일 선택 (.pt)",
                                               filetypes=[("PyTorch 모델", "*.pt"), ("모든 파일", "*.*")])
        if file_path: model_path_var.set(file_path)

    def show_help():
        help_window = tk.Toplevel(root)
        help_window.title("도움말 (Help)")
        help_window.geometry("700x700")

        text_frame = tk.Frame(help_window)
        text_frame.pack(fill="both", expand=True, padx=10, pady=5)
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")
        help_text_widget = tk.Text(text_frame, wrap="word", yscrollcommand=scrollbar.set, padx=5, pady=5, font=("Malgun Gothic", 10))
        help_text_widget.insert("1.0", help_text_content)
        help_text_widget.config(state="disabled")
        help_text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=help_text_widget.yview)

        github_label = tk.Label(help_window, text="개발자 GitHub 방문 (클릭)", fg="blue", cursor="hand2", font=("Malgun Gothic", 10, "underline"))
        github_label.pack(pady=(5,0))
        github_label.bind("<Button-1>", lambda e: webbrowser.open("https://github.com/imonkfcwifi"))

        close_button = tk.Button(help_window, text="닫기", command=help_window.destroy, width=10)
        close_button.pack(pady=10)


    padding_options = {'padx': 10, 'pady': 5}
    file_select_frame = tk.LabelFrame(root, text="파일 선택", padx=10, pady=10)
    file_select_frame.pack(fill='x', **padding_options)

    tk.Label(file_select_frame, text="영상 파일:").grid(row=0, column=0, sticky='w', pady=2)
    tk.Entry(file_select_frame, textvariable=video_path_var, width=70).grid(row=0, column=1, sticky='ew', padx=5, pady=2)
    tk.Button(file_select_frame, text="찾아보기", command=browse_video, width=10).grid(row=0, column=2, padx=(0,5), pady=2)

    tk.Label(file_select_frame, text="모델 파일:").grid(row=1, column=0, sticky='w', pady=2)
    tk.Entry(file_select_frame, textvariable=model_path_var, width=70).grid(row=1, column=1, sticky='ew', padx=5, pady=2)
    tk.Button(file_select_frame, text="찾아보기", command=browse_model, width=10).grid(row=1, column=2, padx=(0,5), pady=2)

    file_select_frame.columnconfigure(1, weight=1)

    # 분석 모드 선택 프레임 추가
    mode_select_frame = tk.LabelFrame(root, text="분석 모드 선택", padx=10, pady=10)
    mode_select_frame.pack(fill='x', **padding_options)
    
    tk.Radiobutton(mode_select_frame, text="호모그래피 모드 (기존 방식) - 4개 점으로 ROI 설정 후 실제 가로/세로 길이 입력", 
                   variable=analysis_mode_var, value="homography", font=("Malgun Gothic", 10)).pack(anchor='w', pady=2)
    tk.Radiobutton(mode_select_frame, text="항공뷰 모드 (드론 촬영용) - 기준선 하나로 스케일 설정", 
                   variable=analysis_mode_var, value="aerial", font=("Malgun Gothic", 10)).pack(anchor='w', pady=2)
    tk.Radiobutton(mode_select_frame, text="주행모드 (운전 중 촬영) - 표지판 검지 및 분석", 
                   variable=analysis_mode_var, value="driving", font=("Malgun Gothic", 10)).pack(anchor='w', pady=2)
    tk.Radiobutton(mode_select_frame, text="테스트 모드 - PT 모델 + 이미지로 검지 성능 테스트", 
                   variable=analysis_mode_var, value="test", font=("Malgun Gothic", 10)).pack(anchor='w', pady=2)

    button_frame = tk.Frame(root)
    button_frame.pack(fill='x', pady=15, padx=10)

    start_button = tk.Button(button_frame, text="분석 시작",
        command=lambda: start_analysis(video_path_var.get(), model_path_var.get(), help_text_content, analysis_mode_var.get()),
        font=("Helvetica", 14, "bold"), bg="#4CAF50", fg="white", relief="raised", width=12, height=1, padx=10, pady=5)
    start_button.pack(side='left', expand=True, padx=5)

    help_button = tk.Button(button_frame, text="도움말", command=show_help,
        font=("Helvetica", 14), bg="#2196F3", fg="white", relief="raised", width=12, height=1, padx=10, pady=5)
    help_button.pack(side='right', expand=True, padx=5)

    root.mainloop()

if __name__ == "__main__":
    
    main()
