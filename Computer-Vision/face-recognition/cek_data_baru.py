# -*- coding: utf-8 -*-
"""
Predict faces on images (file or folder):
- Detector + embedder: InsightFace buffalo_s (SCRFD + MobileFaceNet)
- Classifier: cosine to centroids
- Output: annotated images in outputs/, and outputs/predictions.jsonl

Prereq (in venv):
pip install opencv-contrib-python==4.9.0.80 insightface==0.7.3 onnxruntime==1.18.1 numpy==1.26.4 tqdm
"""

import argparse, json, sys, traceback
from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np
import cv2
from insightface.app import FaceAnalysis

# ====== CONFIG DEFAULTS ======
PACK_NAME     = "buffalo_s"          # ringan & konsisten dgn build_db_light.py
DET_SIZE      = (448, 448)           # kecil → cepat
UNKNOWN_THR   = 0.38                 # turunkan jika sering "Unknown"
ALLOWED       = {".jpg",".jpeg",".png",".bmp",".webp"}
CENT_PATH     = Path("models/face_db_centroids.pkl")
EMB_PATH      = Path("embeddings_buffalo_s.pkl")     # fallback jika centroid belum ada
OUT_DIR       = Path("outputs")

# ====== UTILS ======
def l2n(v, eps=1e-12):
    v = v.astype(np.float32); return v/(np.linalg.norm(v)+eps)

def cosine_sim(a, b):  # a: (D,), b: (N,D)
    return (a @ b.T)

def list_images(path: Path):
    if path.is_file():
        return [path] if path.suffix.lower() in ALLOWED else []
    files = []
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED:
            files.append(p)
    return files

def load_centroids():
    if CENT_PATH.exists():
        db = pickle.loads(CENT_PATH.read_bytes())
        labels = np.array(db["labels"], dtype=object)
        cents  = db["centroids"].astype(np.float32)
        print(f"✅ Loaded centroids: {len(labels)} persons")
        return labels, cents
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
        print(f"⚠️  Built centroids from {EMB_PATH.name}: {len(labels)} persons")
        return np.array(labels, dtype=object), cents
    raise SystemExit("❌ Tidak menemukan centroid atau embeddings. Jalankan build_db_light.py / build_centroids.py dulu.")

def make_app():
    app = FaceAnalysis(name=PACK_NAME)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    det = app.models.get("detection")
    if hasattr(det, "threshold"):
        det.threshold = 0.22
    return app

def annotate_and_save(img_bgr, preds, out_path):
    """
    preds: list of dict {bbox:(x1,y1,x2,y2), name:str, score:float}
    """
    im = img_bgr.copy()
    for p in preds:
        x1,y1,x2,y2 = map(int, p["bbox"])
        name, score = p["name"], float(p["score"])
        is_unk = (name == "Unknown")
        color = (60,60,230) if is_unk else (80,220,80)
        label = f"{name} ({score*100:.1f}%)"
        cv2.rectangle(im, (x1,y1), (x2,y2), color, 2)
        cv2.putText(im, label, (x1, max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), im)

def predict_on_image(app, labels, cents, img_path: Path, thr: float):
    img = cv2.imread(str(img_path))
    if img is None:
        return {"file": str(img_path), "error": "cannot_read_image", "preds": []}

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb)
    if not faces:
        return {"file": str(img_path), "preds": [], "note": "no_face"}

    preds = []
    for f in faces:
        x1,y1,x2,y2 = map(int, f.bbox)
        emb = l2n(f.embedding)
        sims = cosine_sim(emb, cents)   # (N_person,)
        idx  = int(np.argmax(sims))
        score = float(sims[idx])
        name = labels[idx] if score >= thr else "Unknown"
        preds.append({
            "bbox": (x1,y1,x2,y2),
            "name": str(name),
            "score": score
        })
    return {"file": str(img_path), "preds": preds}

def main():
    parser = argparse.ArgumentParser(description="Predict faces on image(s) with InsightFace buffalo_s + centroids")
    parser.add_argument("--img", required=True, help="path to image file or folder (recursively scanned)")
    parser.add_argument("--thr", type=float, default=UNKNOWN_THR, help=f"unknown threshold (default {UNKNOWN_THR})")
    parser.add_argument("--save", action="store_true", help="save annotated images to outputs/")
    args = parser.parse_args()

    src = Path(args.img)
    if not src.exists():
        print(f"❌ Path tidak ditemukan: {src}")
        sys.exit(1)

    labels, cents = load_centroids()
    app = make_app()

    files = list_images(src)
    if not files:
        print("❌ Tidak ada gambar yang cocok.")
        sys.exit(1)

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    log_path = OUT_DIR / "predictions.jsonl"
    log_f = log_path.open("w", encoding="utf-8")

    ok_cnt, nf_cnt = 0, 0
    for fp in tqdm(files, desc="Predicting"):
        try:
            res = predict_on_image(app, labels, cents, fp, args.thr)
            # print to console
            if not res["preds"]:
                nf_cnt += 1
                print(f"[{fp.name}] no face / no preds")
            else:
                print(f"[{fp.name}]")
                for p in res["preds"]:
                    print(f"  - {p['name']:>12s} | {p['score']*100:5.1f}% | bbox={p['bbox']}")
                ok_cnt += 1

            # save annotated image (optional)
            if args.save and res["preds"]:
                sub = fp.relative_to(src) if src.is_dir() else fp.name
                out_img = OUT_DIR / (str(sub) if isinstance(sub, str) else sub.as_posix())
                out_img = out_img if out_img.suffix else out_img.with_suffix(".jpg")
                annotate_and_save(cv2.imread(str(fp)), res["preds"], out_img)

            # write jsonl
            log_f.write(json.dumps(res, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"⚠️ Error on {fp}: {e}")
            traceback.print_exc()
            log_f.write(json.dumps({"file": str(fp), "error": str(e)}) + "\n")

    log_f.close()
    print("\n✅ Selesai.")
    print(f"  • total file        : {len(files)}")
    print(f"  • ada prediksi      : {ok_cnt}")
    print(f"  • tanpa prediksi    : {nf_cnt}")
    if args.save:
        print(f"  • annotated images  : {OUT_DIR}/")
    print(f"  • JSONL log         : {log_path}")

if __name__ == "__main__":
    main()
