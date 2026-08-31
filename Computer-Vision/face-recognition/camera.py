# -*- coding: utf-8 -*-
"""
Ultra-light live face recognition (patched):
- Detector: MediaPipe FaceDetection (CPU cepat)
- Embedder: InsightFace (buffalo_s) via FaceAnalysis.get pada crop 112x112 (aman lintas versi)
- Classifier: cosine to centroids (UNKNOWN_THR)
- Tracking: MOSSE (sangat ringan)
- Kamera: threaded + MJPG @ 640x360
"""

import time, threading, queue, pickle
from pathlib import Path
import numpy as np, cv2
import mediapipe as mp
from insightface.app import FaceAnalysis

# ======== CONFIG ========
UNKNOWN_THR   = 0.38      # turunkan ke 0.36-0.38 jika sering Unknown
CAM_W, CAM_H  = 640, 360  # 640x360 lebih ringan dari 640x480
INFER_SCALE   = 0.60      # deteksi di frame kecil
FRAME_SKIP    = 6         # deteksi tiap N frame (antaranya tracking)
DRAW_LANDMARKS = False    # MediaPipe tidak punya 106 landmark; biarkan False

DB_CENT_PATH  = Path("models/face_db_centroids.pkl")
DB_EMB_PATH   = Path("embeddings_buffalo_s.pkl")   # fallback bila centroid tak ada

# ======== UTILS ========
def l2n(v, eps=1e-12):
    v = v.astype(np.float32); return v/(np.linalg.norm(v)+eps)

def cosine_sim(a,b):  # a:(D,), b:(N,D)
    return (a @ b.T)

def load_centroids():
    if DB_CENT_PATH.exists():
        db = pickle.loads(DB_CENT_PATH.read_bytes())
        labels = db["labels"]; cents = db["centroids"].astype(np.float32)
        print(f"✅ Centroids loaded: {len(labels)} persons"); return labels, cents
    if DB_EMB_PATH.exists():
        db = pickle.loads(DB_EMB_PATH.read_bytes())
        names = np.array(db["names"], dtype=object); embs = db["embs"].astype(np.float32)
        labels = sorted(set(names.tolist()))
        cents = []
        for lab in labels:
            V = embs[names==lab]
            cents.append(l2n(V.mean(0)))
        cents = np.vstack(cents).astype(np.float32)
        print(f"⚠️  Centroids built on the fly from {DB_EMB_PATH.name}: {len(labels)} persons")
        return labels, cents
    raise SystemExit("❌ Tidak ada DB embedding/centroid. Jalankan build_db_light.py &/atau build_centroids.py dulu.")

# ======== Embedder aman (pakai FaceAnalysis.get pada crop 112x112) ========
class SafeEmbedder:
    def __init__(self):
        # det_size kecil → sangat cepat karena cuma memproses crop 112x112
        self.app = FaceAnalysis(name="buffalo_s")
        self.app.prepare(ctx_id=0, det_size=(128,128))
    def get(self, face112_bgr):
        faces = self.app.get(cv2.cvtColor(face112_bgr, cv2.COLOR_BGR2RGB))
        if not faces:
            return None
        return faces[0].embedding  # (512,)

def crop_to_arcface(face_bgr, target=112):
    return cv2.resize(face_bgr, (target, target), interpolation=cv2.INTER_LINEAR)

# ======== MediaPipe detector (CPU cepat) ========
mp_fd = mp.solutions.face_detection
def mediapipe_detect(detector, rgb_small):
    H, W, _ = rgb_small.shape
    res = detector.process(rgb_small)
    boxes = []
    if res.detections:
        for d in res.detections:
            bb = d.location_data.relative_bounding_box
            x1 = int(bb.xmin * W); y1 = int(bb.ymin * H)
            w  = int(bb.width * W); h  = int(bb.height * H)
            # margin kecil agar crop aman
            pad = int(0.15 * max(w,h))
            x1 -= pad; y1 -= pad; w += 2*pad; h += 2*pad
            x2 = x1 + w; y2 = y1 + h
            boxes.append((x1,y1,x2,y2))
    return boxes

# ===== Kamera threaded low-latency (MJPG) =====
class Cam:
    def __init__(self, idx=0, w=CAM_W, h=CAM_H):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened(): raise SystemExit("❌ Kamera tidak bisa dibuka.")
        try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except: pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except: pass
        self.cap, self.q, self.alive = cap, queue.Queue(maxsize=1), True
        threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        while self.alive:
            ok, f = self.cap.read()
            if not ok: continue
            if not self.q.empty():
                try: self.q.get_nowait()
                except: pass
            self.q.put(f)
    def read(self, timeout=1/30):
        try: return True, self.q.get(timeout=timeout)
        except queue.Empty: return False, None
    def release(self):
        self.alive=False; self.cap.release()

# ===== tracker super ringan (MOSSE) -> butuh opencv-contrib-python =====
def new_tracker(bbox, frame):
    trk = cv2.legacy.TrackerMOSSE_create()
    trk.init(frame, tuple(bbox))
    return trk

def main():
    try:
        cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
    except: pass

    labels, cents = load_centroids()
    embedder = SafeEmbedder()
    cam = Cam()

    detector = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.45)

    trackers, metas = [], []   # metas: (name, score)
    k, fcnt = 0, 0
    t0 = time.time()

    print("▶ Kamera ON — tekan 'q' untuk keluar.")
    while True:
        ok, frame = cam.read()
        if not ok or frame is None: continue
        fcnt += 1

        small = cv2.resize(frame, None, fx=INFER_SCALE, fy=INFER_SCALE, interpolation=cv2.INTER_AREA)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        need_det = (k % FRAME_SKIP == 0) or (len(trackers) == 0)
        k += 1

        if need_det:
            boxes_small = mediapipe_detect(detector, rgb_small)
            trackers, metas = [], []
            s = 1.0/INFER_SCALE
            for (x1s,y1s,x2s,y2s) in boxes_small:
                # skala bbox ke frame asli + clamp
                x1 = max(0, int(x1s*s)); y1 = max(0, int(y1s*s))
                x2 = min(frame.shape[1]-1, int(x2s*s)); y2 = min(frame.shape[0]-1, int(y2s*s))
                if x2<=x1 or y2<=y1: continue
                w, h = x2-x1, y2-y1
                face = frame[y1:y2, x1:x2]
                if face.size == 0: continue

                # crop 112x112 -> embedding aman
                face112 = crop_to_arcface(face, 112)
                emb_vec = embedder.get(face112)
                if emb_vec is None: 
                    continue
                emb = l2n(emb_vec)

                # cosine ke centroid
                sim = cosine_sim(emb, cents)  # (N,)
                idx = int(np.argmax(sim)); score = float(sim[idx])
                name = labels[idx] if score >= UNKNOWN_THR else "Unknown"

                trk = new_tracker((x1,y1,w,h), frame)
                trackers.append(trk); metas.append((name, score))
        else:
            new_trk, new_meta = [], []
            for trk, meta in zip(trackers, metas):
                ok, box = trk.update(frame)
                if not ok: continue
                new_trk.append(trk); new_meta.append(meta)
            trackers, metas = new_trk, new_meta

        # render
        for trk, (name, score) in zip(trackers, metas):
            ok, (x,y,w,h) = trk.update(frame)
            if not ok: continue
            x2, y2 = x+w, y+h
            is_unk = score < UNKNOWN_THR
            label = f"{'Unknown' if is_unk else name} ({score*100:.1f}%)"
            color = (60,60,230) if is_unk else (80,220,80)
            cv2.rectangle(frame,(int(x),int(y)),(int(x2),int(y2)), color, 2)
            cv2.putText(frame,label,(int(x),max(22,int(y)-8)),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2,cv2.LINE_AA)

        # FPS overlay tiap 20 frame
        if fcnt % 20 == 0:
            fps = fcnt / max(time.time()-t0, 1e-6)
            cv2.putText(frame, f"{fps:.1f} FPS (MP+ArcFace)", (10,24),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(50,200,255),2)

        cv2.imshow("Face Attendance — ULTRALIGHT (MediaPipe+ArcFace)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cam.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
