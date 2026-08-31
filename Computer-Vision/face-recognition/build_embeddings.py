# build_db_light.py
import pickle
from pathlib import Path
import numpy as np, cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis

ROOT = Path("DATASET SISTEM PRESENSI WAJAH V2.0")
TRAIN_DIR = ROOT / "Data Train"
OUT_PATH = Path("embeddings_buffalo_s.pkl")
ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def l2n(v, eps=1e-12): 
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + eps)

def list_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED]

def extract_label_from_filename(p: Path) -> str | None:
    s = p.stem
    parts = s.split("_")
    if len(parts) < 3: return None
    return parts[1].strip().title()

def main():
    assert TRAIN_DIR.is_dir(), f"Folder tidak ditemukan: {TRAIN_DIR}"
    files = list_images(TRAIN_DIR)
    print(f"[INFO] Train files: {len(files)}")

    app = FaceAnalysis(name="buffalo_s")  # <— model ringan
    app.prepare(ctx_id=0, det_size=(448, 448))

    names, embs = [], []
    for fp in tqdm(files, desc="Build embeddings"):
        label = extract_label_from_filename(fp)
        if not label:
            continue
        img = cv2.imread(str(fp))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)
        if not faces:
            continue
        f = max(faces, key=lambda a: (a.bbox[2]-a.bbox[0])*(a.bbox[3]-a.bbox[1]))
        emb = l2n(f.embedding)
        embs.append(emb)
        names.append(label)

    if not embs:
        raise SystemExit("❌ Tidak ada embedding. Pastikan wajah jelas & format file sesuai 523xxx_Nama_XX.jpg")

    data = {"embs": np.vstack(embs), "names": np.array(names, dtype=object)}
    with open(OUT_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"✅ Saved {OUT_PATH} | embeddings: {data['embs'].shape} | persons: {len(set(names))}")

if __name__ == "__main__":
    main()
