from pathlib import Path
import pickle, json
import numpy as np, cv2, sys
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# ====== KONFIGURASI ======
ROOT = Path("DATASET SISTEM PRESENSI WAJAH V2.0")
TEST_DIR = ROOT / "Data Test"           # folder uji
EMB_PATH = Path("embeddings_buffalo_s.pkl")     # hasil build_db_light.py
CENT_PATH = Path("models/face_db_centroids.pkl")  # opsional (lebih cepat)
PACK_NAME = "buffalo_s"                  # model ringan
DET_SIZE  = (448, 448)                   # bisa turunkan lagi kalau CPU lemah
UNKNOWN_THR = 0.38                       # turunkan jika sering "Unknown"
ALLOWED = {".jpg",".jpeg",".png",".bmp",".webp"}

# ====== UTIL ======
def l2n(v, eps=1e-12):
    v = v.astype(np.float32); return v/(np.linalg.norm(v)+eps)

def list_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED]

def label_from_filename(p: Path):
    """Format nama file: 523xxx_Nama Orang_XX.jpg -> ambil 'Nama Orang'."""
    s = p.stem
    parts = s.split("_")
    if len(parts) < 3: return None
    return parts[1].strip().title()

def load_centroids():
    # 1) kalau file centroid ada → pakai
    if CENT_PATH.exists():
        db = pickle.loads(CENT_PATH.read_bytes())
        return np.array(db["labels"], dtype=object), db["centroids"].astype(np.float32)
    # 2) kalau belum ada → bangun dari file embedding
    if EMB_PATH.exists():
        db = pickle.loads(EMB_PATH.read_bytes())
        names = np.array(db["names"], dtype=object)
        embs  = db["embs"].astype(np.float32)
        labels = sorted(set(names.tolist()))
        cents = []
        for lab in labels:
            V = embs[names == lab]
            cents.append(l2n(V.mean(0)))
        cents = np.vstack(cents).astype(np.float32)
        return np.array(labels, dtype=object), cents
    raise SystemExit("❌ Tidak menemukan CENTROID maupun EMBEDDING. Jalankan build_db_light.py dulu.")

# ====== MODEL ======
def load_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name=PACK_NAME)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    det = app.models.get("detection")
    if hasattr(det, "threshold"): det.threshold = 0.22
    return app

# ====== EVALUASI ======
def main():
    assert TEST_DIR.is_dir(), f"Folder test tidak ditemukan: {TEST_DIR}"

    labels, cents = load_centroids()
    label_to_idx = {lab:i for i,lab in enumerate(labels)}
    app = load_app()

    img_files = list_images(TEST_DIR)
    if not img_files:
        raise SystemExit("❌ Tidak ada gambar di Data Test.")

    y_true, y_pred = [], []
    n_nf, n_unk = 0, 0  # no-face, predicted unknown
    pbar = tqdm(img_files, desc="Evaluating")

    for fp in pbar:
        gt = label_from_filename(fp)
        if gt is None:    # skip file yang format namanya tidak sesuai
            continue

        img = cv2.imread(str(fp))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = app.get(rgb)

        if not faces:
            n_nf += 1
            # anggap prediksi salah → masukkan 'Unknown'
            y_true.append(gt); y_pred.append("Unknown")
            continue

        # ambil wajah terbesar
        f = max(faces, key=lambda a: (a.bbox[2]-a.bbox[0])*(a.bbox[3]-a.bbox[1]))
        emb = l2n(f.embedding)

        # cosine similarity ke semua centroid
        sims = emb @ cents.T                   # (N_person,)
        idx  = int(np.argmax(sims))
        score = float(sims[idx])
        pred = labels[idx] if score >= UNKNOWN_THR else "Unknown"
        if pred == "Unknown": n_unk += 1

        y_true.append(gt); y_pred.append(pred)

    # ====== METRIK ======
    y_true = np.array(y_true, dtype=object)
    y_pred = np.array(y_pred, dtype=object)
    valid_mask = np.isin(y_true, labels)      # filter hanya kelas yang ada di centroid
    yt, yp = y_true[valid_mask], y_pred[valid_mask]

    total = len(yt)
    correct = int(np.sum(yt == yp))
    acc = (correct / total * 100.0) if total else 0.0

    print("\n================= HASIL EVALUASI =================")
    print(f"Total sampel tervalidasi : {total}")
    print(f"Benar (top-1)            : {correct}")
    print(f"Akurasi                  : {acc:.2f}%")
    print(f"Unknown rate             : {n_unk} dari {len(img_files)} file")
    print(f"No-face (deteksi gagal)  : {n_nf} dari {len(img_files)} file")

    # Report per-class (hanya kelas yang muncul)
    present_classes = sorted(set(yt.tolist()))
    try:
        from sklearn.metrics import classification_report, ConfusionMatrixDisplay
        print("\nClassification report:")
        print(classification_report(yt, yp, labels=present_classes, zero_division=0))
    except Exception as e:
        print("Classification report gagal dibuat:", e)

    # Confusion matrix opsional disimpan ke file
    try:
        cm = confusion_matrix(yt, yp, labels=present_classes)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,8))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        ax.set_xticks(range(len(present_classes)), present_classes, rotation=90)
        ax.set_yticks(range(len(present_classes)), present_classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix (Acc {acc:.2f}%)")
        plt.tight_layout()
        fig.savefig("confusion_matrix.png", dpi=160)
        plt.close(fig)
        print("✅ Confusion matrix disimpan ke: confusion_matrix.png")
    except Exception as e:
        print("Confusion matrix gagal dibuat:", e)

if __name__ == "__main__":
    main() 