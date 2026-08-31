# camera_capture_predict.py
# -*- coding: utf-8 -*-
"""
Live camera + capture prediction:
- Kamera live ditampilkan
- Tekan 'c' untuk capture -> deteksi & kenali wajah (nama/Unknown + skor)
- Hasil capture muncul di jendela terpisah
"""

import time, threading, queue, pickle
from pathlib import Path
import numpy as np, cv2
from insightface.app import FaceAnalysis

# ===== CONFIG =====
PACK_NAME   = "buffalo_s"
DET_SIZE    = (448, 448)
UNKNOWN_THR = 0.38
CAM_W, CAM_H = 640, 360

CENT_PATH   = Path("models/face_db_centroids.pkl")
EMB_PATH    = Path("embeddings_buffalo_s.pkl")
OUT_DIR     = Path("outputs_captures")

def l2n(v, eps=1e-12):
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + eps)

def cosine_sim(a, b):
    return (a @ b.T)

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
        print(f"⚠️ Built centroids from {EMB_PATH.name}: {len(labels)} persons")
        return np.array(labels, dtype=object), cents
    raise SystemExit("❌ Tidak ada centroid/embeddings DB. Jalankan training dulu.")

def make_app():
    app = FaceAnalysis(name=PACK_NAME)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    det = app.models.get("detection")
    if hasattr(det, "threshold"):
        det.threshold = 0.22
    return app

class Cam:
    def __init__(self, idx=0, w=CAM_W, h=CAM_H):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            raise SystemExit("❌ Kamera tidak bisa dibuka.")
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass
        self.cap, self.q, self.alive = cap, queue.Queue(maxsize=1), True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.alive:
            ok, f = self.cap.read()
            if not ok:
                continue
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except:
                    pass
            self.q.put(f)

    def read(self, timeout=1/30):
        try:
            return True, self.q.get(timeout=timeout)
        except queue.Empty:
            return False, None

    def release(self):
        self.alive = False
        self.cap.release()

def annotate(img_bgr, preds):
    im = img_bgr.copy()
    for p in preds:
        x1, y1, x2, y2 = map(int, p["bbox"])
        name, score = p["name"], float(p["score"])
        is_unk = (name == "Unknown")
        color = (60, 60, 230) if is_unk else (80, 220, 80)
        label = f"{name} ({score*100:.1f}%)"
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
        cv2.putText(im, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return im

def predict_frame(app, labels, cents, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb)
    preds = []
    if not faces:
        return preds
    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        emb = l2n(f.embedding)
        sims = cosine_sim(emb, cents)
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        name = labels[idx] if score >= UNKNOWN_THR else "Unknown"
        preds.append({"bbox": (x1, y1, x2, y2), "name": str(name), "score": score})
    return preds

def main():
    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except:
        pass

    labels, cents = load_centroids()
    app = make_app()
    cam = Cam()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("▶ Kamera ON")
    print("   • Tekan 'c' untuk capture & prediksi")
    print("   • Tekan 'q' untuk keluar\n")

    capture_idx = 0

    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            continue

        cv2.putText(frame, "Press 'c' to capture, 'q' to quit",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 255), 2)
        cv2.imshow("Live Camera", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == ord('c'):
            # freeze current frame & run prediction
            preds = predict_frame(app, labels, cents, frame)
            annotated = annotate(frame, preds) if preds else frame.copy()

            # tampilkan hasil capture
            win_name = "Capture Result"
            cv2.imshow(win_name, annotated)

            # simpan ke file
            out_path = OUT_DIR / f"capture_{capture_idx:03d}.jpg"
            cv2.imwrite(str(out_path), annotated)
            print(f"📸 Capture disimpan: {out_path}")
            for p in preds:
                print(f"  -> {p['name']} ({p['score']*100:.1f}%) bbox={p['bbox']}")
            capture_idx += 1

            # tunggu user tekan tombol untuk tutup window hasil (tanpa stop kamera utama)
            cv2.waitKey(1)

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()