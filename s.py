import os, json, shutil, re, random, ast
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import GroupShuffleSplit

# ---------------------------
# 변환/분할 로직
# ---------------------------

def group_key_from_name(name: str, pattern: str):
    base = Path(name).stem
    if not pattern:
        return base.split('_')[0]
    try:
        m = re.compile(pattern).match(base)
        return m.group(1) if m else base.split('_')[0]
    except re.error:
        return base.split('_')[0]

def convert_one(args):
    json_path, img_dir, yolo_out_dir, class_map, group_regex = args
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        img_name = data["image"]["filename"]
        W, H = data["image"]["imsize"]  # [W, H]
        ann = data.get("annotation", [])

        lines = []
        for obj in ann:
            cls_name = obj.get("class")
            if cls_name not in class_map:
                continue
            cls_id = class_map[cls_name]
            box = obj.get("box")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = box
            xc = ((x1 + x2) / 2.0) / W
            yc = ((y1 + y2) / 2.0) / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        img_path = Path(img_dir) / img_name
        if lines and img_path.exists():
            out_txt = Path(yolo_out_dir) / (Path(img_name).stem + ".txt")
            out_txt.write_text("\n".join(lines), encoding="utf-8")
            return {
                "img": str(img_path.resolve()),
                "label_txt": str(out_txt.resolve()),
                "group": group_key_from_name(img_name, group_regex),
                "has_label": True
            }
        else:
            return {
                "img": str(img_path.resolve()),
                "label_txt": None,
                "group": group_key_from_name(img_name, group_regex),
                "has_label": False
            }
    except Exception as e:
        return {"error": f"{Path(json_path).name}: {e}"}

def run_pipeline(images_dir, out_root, val_ratio, seed,
                 class_map_str, group_regex, log_fn,
                 json_files_override):

    random.seed(seed)
    images_dir = Path(images_dir)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 출력 폴더
    yolo_all = out_root / "labels_yolo_all"
    yolo_all.mkdir(exist_ok=True)
    images_out_train = out_root / "images/train"
    images_out_val = out_root / "images/val"
    labels_out_train = out_root / "labels/train"
    labels_out_val = out_root / "labels/val"
    for p in [images_out_train, images_out_val, labels_out_train, labels_out_val]:
        p.mkdir(parents=True, exist_ok=True)

    # 클래스 매핑 파싱
    try:
        clean = (class_map_str
                 .replace("“", '"').replace("”", '"')
                 .replace("‘", "'").replace("’", "'"))
        try:
            class_map = json.loads(clean)
        except:
            class_map = ast.literal_eval(clean)
        class_map = {str(k): int(v) for k, v in class_map.items()}
    except Exception as e:
        raise ValueError(f"클래스 매핑 JSON 파싱 실패: {e}")

    json_files = json_files_override
    if not json_files:
        raise FileNotFoundError("선택된 JSON 파일이 없습니다.")

    log_fn(f"총 JSON 파일: {len(json_files)}개\n")
    log_fn("JSON → YOLO 변환 중...\n")

    tasks = [(jp, str(images_dir), str(yolo_all), class_map, group_regex) for jp in json_files]
    with Pool(processes=max(1, cpu_count()-1)) as pool:
        results = list(pool.map(convert_one, tasks))

    errs = [r["error"] for r in results if r and r.get("error")]
    if errs:
        log_fn(f"변환 에러 {len(errs)}건 (상위 5개)\n")
        for e in errs[:5]: log_fn("  - " + e + "\n")

    rows = [r for r in results if r and not r.get("error")]
    rows = [r for r in rows if r.get("has_label") and r.get("img") and os.path.exists(r["img"])]

    if not rows:
        raise RuntimeError("유효한 라벨/이미지 쌍이 없습니다.")

    log_fn(f"라벨 생성 완료: {len(rows)}장\n")

    # 그룹 기반 split
    groups = [r["group"] for r in rows]
    idxs = list(range(len(rows)))
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(gss.split(idxs, groups=groups))
    train_list = [rows[i] for i in train_idx]
    val_list   = [rows[i] for i in val_idx]

    # 이미지/라벨 복사
    for r in train_list:
        if r["label_txt"]:
            dst_label = labels_out_train / Path(r["label_txt"]).name
            shutil.copy2(r["label_txt"], dst_label)
            dst_img = images_out_train / Path(r["img"]).name
            shutil.copy2(r["img"], dst_img)

    for r in val_list:
        if r["label_txt"]:
            dst_label = labels_out_val / Path(r["label_txt"]).name
            shutil.copy2(r["label_txt"], dst_label)
            dst_img = images_out_val / Path(r["img"]).name
            shutil.copy2(r["img"], dst_img)

    log_fn(f"Split 완료 → train:{len(train_list)} / val:{len(val_list)}\n")

    # data.yaml
    names_sorted = sorted(class_map.items(), key=lambda kv: kv[1])
    names_dict = {int(v): str(k) for k, v in names_sorted}
    yaml_lines = []
    yaml_lines.append(f"path: {out_root.as_posix()}")
    yaml_lines.append(f"train: {images_out_train.as_posix()}")
    yaml_lines.append(f"val: {images_out_val.as_posix()}")
    yaml_lines.append("names:")
    for idx in sorted(names_dict.keys()):
        yaml_lines.append(f"  {idx}: {names_dict[idx]}")
    (out_root / "data.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")

    log_fn("\n✅ 준비 완료!\n")
    log_fn(f"- data.yaml: {out_root / 'data.yaml'}\n")

# ---------------------------
# GUI
# ---------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Dataset Builder (전통구조)")
        self.geometry("780x600")

        self.images_dir = tk.StringVar()
        self.json_files = []
        self.out_root = tk.StringVar()
        self.val_ratio = tk.DoubleVar(value=0.2)
        self.seed = tk.IntVar(value=42)
        self.class_map = tk.StringVar(value='{"traffic_sign": 0}')
        self.group_regex = tk.StringVar(value=r"^([a-zA-Z]*\d+)")

        pad = {"padx": 8, "pady": 6}
        frm = ttk.Frame(self); frm.pack(fill="both", expand=True, **pad)

        ttk.Label(frm, text="원천 이미지 폴더").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.images_dir, width=70).grid(row=0, column=1, sticky="we")
        ttk.Button(frm, text="찾기", command=self.pick_images).grid(row=0, column=2)

        ttk.Label(frm, text="라벨 JSON 파일들").grid(row=1, column=0, sticky="w")
        ttk.Button(frm, text="파일 선택", command=self.pick_json_files).grid(row=1, column=1, sticky="w")
        self.json_label = ttk.Label(frm, text="0개 선택됨"); self.json_label.grid(row=1, column=2, sticky="w")

        ttk.Label(frm, text="출력 루트 폴더").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.out_root, width=70).grid(row=2, column=1, sticky="we")
        ttk.Button(frm, text="찾기", command=self.pick_out).grid(row=2, column=2)

        ttk.Label(frm, text="클래스 매핑(JSON)").grid(row=3, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.class_map, width=70).grid(row=3, column=1, sticky="we", columnspan=2)

        ttk.Button(frm, text="실행", command=self.on_run).grid(row=4, column=2, sticky="e", pady=(8,0))

        ttk.Label(frm, text="로그").grid(row=5, column=0, sticky="w", pady=(10,0))
        self.log = tk.Text(frm, height=20); self.log.grid(row=6, column=0, columnspan=3, sticky="nsew")
        frm.rowconfigure(6, weight=1); frm.columnconfigure(1, weight=1)

    def pick_images(self):
        d = filedialog.askdirectory(title="원천 이미지 폴더 선택")
        if d: self.images_dir.set(d)

    def pick_json_files(self):
        files = filedialog.askopenfilenames(title="라벨 JSON 파일 선택", filetypes=[("JSON files","*.json")])
        if files:
            self.json_files = list(files)
            self.json_label.config(text=f"{len(files)}개 선택됨")

    def pick_out(self):
        d = filedialog.askdirectory(title="출력 루트 폴더 선택")
        if d: self.out_root.set(d)

    def append_log(self, msg: str):
        self.log.insert("end", msg); self.log.see("end"); self.update_idletasks()

    def on_run(self):
        if not self.json_files:
            messagebox.showwarning("확인", "JSON 파일을 하나 이상 선택하세요.")
            return
        if not self.images_dir.get() or not self.out_root.get():
            messagebox.showwarning("확인", "이미지 폴더와 출력 폴더를 지정하세요.")
            return
        self.log.delete("1.0","end")
        self.append_log("=== YOLO Dataset Builder 시작 ===\n")
        try:
            run_pipeline(
                images_dir=self.images_dir.get(),
                out_root=self.out_root.get(),
                val_ratio=float(self.val_ratio.get()),
                seed=int(self.seed.get()),
                class_map_str=self.class_map.get(),
                group_regex=self.group_regex.get(),
                log_fn=self.append_log,
                json_files_override=self.json_files
            )
            self.append_log("\n완료!\n")
            messagebox.showinfo("완료", "데이터셋 구성이 완료되었습니다.")
        except Exception as e:
            self.append_log(f"\n[에러] {e}\n")
            messagebox.showerror("에러", str(e))

if __name__ == "__main__":
    App().mainloop()
