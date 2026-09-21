"""
verifier.py — deteksi + crop + verifikasi kukang (CEPAT).

Dipakai untuk dua hal:
  1. Verifikasi tahap dua sortir: crop objek lalu cek 'kukang / bukan_kukang'.
  2. Menyediakan crop untuk model behavior (crop-based) lewat detect_and_crop().

Detektor: yolo11n @ 416 (jauh lebih ringan di CPU daripada yolo11x@640; pada
uji camera-trap tetap menemukan hewan dengan baik). Ganti ke yolo11x.pt bila
butuh deteksi paling teliti (tapi lambat).
"""
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from ultralytics import YOLO

MODELS_DIR = Path(__file__).parent / "models"
DETECTOR_WEIGHTS = MODELS_DIR / "yolo11n.pt"        # CEPAT (ganti yolo11x.pt utk teliti)
VERIFIER_WEIGHTS = MODELS_DIR / "verifier_best.pt"  # kukang / bukan_kukang

VERIFY_IMGSZ = 224
DETECT_IMGSZ = 416          # 640 lebih teliti tapi ~2x lebih lambat
MIN_CONF = 0.15
MIN_AREA = 0.008
PAD = 0.06

ANIMALISH = {"cat", "dog", "bear", "teddy bear", "sheep", "horse", "cow",
             "elephant", "bird", "zebra", "giraffe", "person"}

_detector = None
_verifier = None


def load_verifier() -> Dict:
    global _detector, _verifier
    if DETECTOR_WEIGHTS.exists():
        _detector = YOLO(str(DETECTOR_WEIGHTS))
    if VERIFIER_WEIGHTS.exists():
        _verifier = YOLO(str(VERIFIER_WEIGHTS))
    return {"detektor": _detector is not None, "verifier": _verifier is not None}


def available() -> bool:
    return _detector is not None and _verifier is not None


def detector_ready() -> bool:
    return _detector is not None


def _best_box(res, W, H):
    if not len(res.boxes):
        return None
    best = None
    for b, c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(),
                       res.boxes.cls.tolist()):
        area = (b[2] - b[0]) * (b[3] - b[1]) / (W * H)
        if area < MIN_AREA:
            continue
        name = res.names[int(k)]
        bonus = 0.15 if name in ANIMALISH else 0.0
        score = c + bonus
        if best is None or score > best[0]:
            best = (score, b)
    return best[1] if best else None


def detect_and_crop(img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
    """Deteksi hewan lalu crop (dengan padding). Kembalikan (crop, True) atau
    (None, False). Dipakai bersama oleh verifikasi & behavior."""
    if _detector is None:
        return None, False
    H, W = img_bgr.shape[:2]
    det = _detector.predict(img_bgr, imgsz=DETECT_IMGSZ, conf=MIN_CONF, verbose=False)[0]
    box = _best_box(det, W, H)
    if box is None:
        return None, False
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * PAD, (y2 - y1) * PAD
    x1 = max(0, int(x1 - pw)); y1 = max(0, int(y1 - ph))
    x2 = min(W, int(x2 + pw)); y2 = min(H, int(y2 + ph))
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, False
    return crop, True


def verify_crop(crop_bgr: np.ndarray) -> Dict:
    """Verifier pada crop yang SUDAH ada (hemat: tak deteksi ulang)."""
    if _verifier is None:
        return {"status": "nonaktif", "is_kukang": None, "confidence": 0.0, "prob": {}}
    res = _verifier.predict(crop_bgr, imgsz=VERIFY_IMGSZ, verbose=False)[0]
    names = res.names
    probs = res.probs.data.tolist()
    prob_map = {names[i]: float(probs[i]) for i in range(len(probs))}
    top = max(prob_map, key=prob_map.get)
    return {
        "status": "ok",
        "is_kukang": (top == "kukang"),
        "confidence": round(prob_map[top], 4),
        "prob": {k: round(v, 4) for k, v in prob_map.items()},
    }


def verify_is_kukang(img_bgr: np.ndarray) -> Dict:
    """Full: deteksi+crop lalu verifikasi (bila crop belum ada)."""
    if not available():
        return {"status": "nonaktif", "is_kukang": None, "confidence": 0.0, "prob": {}}
    crop, ok = detect_and_crop(img_bgr)
    if not ok:
        return {"status": "tak_terdeteksi", "is_kukang": None, "confidence": 0.0, "prob": {}}
    return verify_crop(crop)
