# predict_gui_import.py
# -*- coding: utf-8 -*-
"""
GUI sederhana untuk:
1) Import 1 foto -> prediksi & tampil di jendela
2) Import folder foto -> prediksi semua foto, tampil satu per satu

Model:
- Detector + embedder : InsightFace buffalo_s (CNN)
- Classifier          : cosine similarity ke centroid embedding
"""

import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import pickle
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ================== CONFIG ==================
UNKNOWN_THR = 0.38
PACK_NAME   = "buffalo_s"
DET_SIZE    = (448, 448)

CENT_PATH   = Path("models/face_db_centroids.pkl")
EMB_PATH    = Path("embeddings_buffalo_s.pkl")

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ================== UTIL ====================
def l2n(v, eps=1e-12):
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + eps)


def cosine_sim(a, b):
    # a: (D,), b: (N,D)
    return (a @ b.T)


def resize_keep_ratio(img, max_w=1000, max_h=750):
    """
    Mengecilkan gambar kalau terlalu besar,
    tetap proporsional (tidak melebar/ketarik).
    Tidak pernah memperbesar gambar kecil.
    """
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)  # hanya perkecil
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def show_image(win_name, img):
    """
    Tampilkan gambar dengan ukuran sudah dikecilkan bila perlu.
    Window auto-size (tidak stretch).
    """
    img_disp = resize_keep_ratio(img)
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win_name, img_disp)


def list_images_in_folder(folder: Path):
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
            files.append(p)
    files.sort()
    return files


# ================== LOAD DB =================
def load_centroids():
    # 1) coba load dari face_db_centroids.pkl
    if CENT_PATH.exists():
        db = pickle.loads(CENT_PATH.read_bytes())
        labels = np.array(db["labels"], dtype=object)
        cents  = db["centroids"].astype(np.float32)
        print(f"✅ Loaded centroids: {len(labels)} persons")
        return labels, cents

    # 2) fallback: bangun centroid dari embeddings_buffalo_s.pkl
    if EMB_PATH.exists():
        db    = pickle.loads(EMB_PATH.read_bytes())
        names = np.array(db["names"], dtype=object)
        embs  = db["embs"].astype(np.float32)

        uniq_labels = sorted(set(names.tolist()))
        cents = []
        for lab in uniq_labels:
            V = embs[names == lab]
            cents.append(l2n(V.mean(0)))
        cents = np.vstack(cents).astype(np.float32)
        print(f"⚠️ Built centroids from {EMB_PATH.name}: {len(uniq_labels)} persons")
        return np.array(uniq_labels, dtype=object), cents

    raise SystemExit(
        "❌ Tidak ada 'models/face_db_centroids.pkl' maupun "
        "'embeddings_buffalo_s.pkl'. Jalankan script training / build_embeddings dulu."
    )


def make_app():
    app = FaceAnalysis(name=PACK_NAME)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    det = app.models.get("detection")
    if hasattr(det, "threshold"):
        det.threshold = 0.22  # bisa kamu tweak
    return app


# ================== PREDIKSI =================
def predict_image(img_path: Path, app, labels, cents):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️ Gagal baca gambar: {img_path}")
        return None, []

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb)

    preds = []
    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        emb   = l2n(f.embedding)
        sims  = cosine_sim(emb, cents)
        idx   = int(np.argmax(sims))
        score = float(sims[idx])
        name  = labels[idx] if score >= UNKNOWN_THR else "Unknown"
        preds.append((name, score, (x1, y1, x2, y2)))

    return img, preds


def annotate(img, preds):
    im = img.copy()
    if not preds:
        cv2.putText(
            im, "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA
        )
        return im

    for (name, score, (x1, y1, x2, y2)) in preds:
        is_unk = (name == "Unknown")
        color = (60, 60, 230) if is_unk else (80, 220, 80)
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
        label = f"{name} ({score*100:.1f}%)"
        cv2.putText(
            im, label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
        )
    return im


# ================== GUI CALLBACKS =============
def on_import_photo():
    path_str = filedialog.askopenfilename(
        title="Pilih Foto",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*"),
        ],
    )
    if not path_str:
        return

    img_path = Path(path_str)
    img, preds = predict_image(img_path, app, labels, cents)
    if img is None:
        return

    annotated = annotate(img, preds)

    print(f"\n[Foto] {img_path.name}")
    for (name, score, _) in preds:
        print(f"  -> {name} ({score*100:.1f}%)")

    show_image("Hasil Prediksi", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def on_import_folder():
    folder_str = filedialog.askdirectory(title="Pilih Folder Berisi Foto")
    if not folder_str:
        return

    folder = Path(folder_str)
    files = list_images_in_folder(folder)
    if not files:
        print("⚠️ Tidak ada gambar di folder itu.")
        return

    print("\n=== MODE FOLDER ===")
    print("  • Spasi / Enter / N : next foto")
    print("  • Q / Esc           : keluar mode folder\n")

    for img_path in files:
        img, preds = predict_image(img_path, app, labels, cents)
        if img is None:
            continue

        annotated = annotate(img, preds)
        print(f"[Foto] {img_path.relative_to(folder)}")
        for (name, score, _) in preds:
            print(f"  -> {name} ({score*100:.1f}%)")

        show_image("Hasil Prediksi (Folder)", annotated)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord(" "), ord("n"), ord("N"), 13):  # space / n / enter
                break
            if key in (ord("q"), ord("Q"), 27):  # q / esc
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()


# ================== MAIN ======================
if __name__ == "__main__":
    # kurangi lag
    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except:
        pass

    # load DB & model sekali di awal
    labels, cents = load_centroids()
    app = make_app()

    # ---- GUI Tkinter ----
    root = tk.Tk()
    root.title("Face Recognition (CNN) - Import Foto / Folder")
    root.geometry("430x230")

    title_lbl = tk.Label(
        root,
        text="Aplikasi Pengenalan Wajah (CNN - buffalo_s)",
        font=("Segoe UI", 11),
    )
    title_lbl.pack(pady=10)

    btn1 = tk.Button(
        root,
        text="📷 Import 1 Foto",
        font=("Segoe UI", 14),
        width=20,
        command=on_import_photo,
    )
    btn1.pack(pady=10)

    btn2 = tk.Button(
        root,
        text="🗂️  Import Folder Foto",
        font=("Segoe UI", 14),
        width=20,
        command=on_import_folder,
    )
    btn2.pack(pady=5)

    note_lbl = tk.Label(
        root,
        text="Foto dikenali berdasarkan DB embedding yang sudah kamu latih.",
        font=("Segoe UI", 9),
    )
    note_lbl.pack(pady=10)

    root.mainloop()
