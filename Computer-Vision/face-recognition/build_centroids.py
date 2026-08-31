# -*- coding: utf-8 -*-
import pickle
from pathlib import Path
import numpy as np, os

EMB_PATH   = Path("embeddings_buffalo_s.pkl")   # <-- ganti kalau nama lain
OUT_PATH   = Path("models/face_db_centroids.pkl")

def l2n(v, eps=1e-12):
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + eps)

def main():
    assert EMB_PATH.exists(), f"File embedding tidak ditemukan: {EMB_PATH}"
    os.makedirs(OUT_PATH.parent, exist_ok=True)

    db = pickle.loads(EMB_PATH.read_bytes())
    names = np.array(db["names"], dtype=object)
    embs  = db["embs"].astype(np.float32)

    labels = sorted(set(names.tolist()))
    cents = []
    for lab in labels:
        V = embs[names == lab]
        c = l2n(V.mean(0))
        cents.append(c)
    cents = np.vstack(cents).astype(np.float32)

    out = {"labels": labels, "centroids": cents}
    OUT_PATH.write_bytes(pickle.dumps(out))
    print(f"✅ Saved {OUT_PATH} | persons={len(labels)} | dim={cents.shape[1]}")
    print("Siap dipakai kamera.")

if __name__ == "__main__":
    main()