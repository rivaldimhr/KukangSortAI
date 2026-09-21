"""
inference.py — memuat dua model YOLO11n-cls dan menjalankan prediksi.

  MODEL 1 (sortir)   : kelas kukang / unknown / delete
  MODEL 2 (behavior) : kelas bergerak / grooming / makan

Preprocessing enhance() diterapkan sebelum predict, PERSIS seperti training.
Model dimuat sekali saat startup, bukan tiap request.
"""
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from ultralytics import YOLO

from preprocessing import enhance, blur_score, blur_label
import verifier  # tahap verifikasi kedua (crop + cek kukang)

MODELS_DIR = Path(__file__).parent / "models"

# Ganti nama berkas ini sesuai bobotmu.
SORT_WEIGHTS = MODELS_DIR / "sortir_best.pt"
BEHAVIOR_WEIGHTS = MODELS_DIR / "kukang_behaviour_best.pt"     # TAHAP 1: FULL image (enhance, 320)
BEHAVIOR_CROP_WEIGHTS = MODELS_DIR / "behaviour_best_crop.pt"  # TAHAP 2: CROP (224)

# Urutan kelas mengikuti model.names asli:
#   sortir   : {0: delete, 1: kukang, 2: unknown}
#   behavior : {0: bergerak, 1: grooming, 2: makan}
# Kode di bawah memetakan berdasarkan NAMA, jadi urutan daftar ini
# hanya menentukan urutan tampil di frontend, bukan kebenaran label.
SORT_CLASSES = ["kukang", "unknown", "delete"]
BEHAVIOR_CLASSES = ["bergerak", "grooming", "makan"]

CONFIDENCE_THRESHOLD = 0.70  # sama dengan notebook Step 9
IMGSZ = 224                   # crop behavior & sortir
FULL_IMGSZ = 320             # behavior FULL image (sesuai notebook full)

# Verifikasi tahap dua: hanya dijalankan saat hasil sortir = 'kukang'.
VERIFY_ENABLED = True
# Hanya turunkan kukang->unknown bila verifier SANGAT yakin bukan_kukang.
# Ambang tinggi melindungi kukang asli dari crop detektor yang meleset.
VERIFY_DEMOTE_THRESHOLD = 0.85

# Behavior dua-tahap (cek ulang 2 model crop). Kalau kedua model TIDAK sepakat,
# atau keyakinan di bawah ambang, atau hewan tak terdeteksi -> perlu_review.
BEHAVIOR_REVIEW_THRESHOLD = 0.60

_sort_model = None
_behavior_model = None
_behavior_crop_model = None


def load_models():
    """Dipanggil sekali saat startup FastAPI."""
    global _sort_model, _behavior_model, _behavior_crop_model
    if SORT_WEIGHTS.exists():
        _sort_model = YOLO(str(SORT_WEIGHTS))
    if BEHAVIOR_WEIGHTS.exists():
        _behavior_model = YOLO(str(BEHAVIOR_WEIGHTS))
    if BEHAVIOR_CROP_WEIGHTS.exists():
        _behavior_crop_model = YOLO(str(BEHAVIOR_CROP_WEIGHTS))
    vstat = verifier.load_verifier()
    return {
        "sortir": _sort_model is not None,
        "behavior": _behavior_model is not None,
        "behavior_crop": _behavior_crop_model is not None,
        "detektor": vstat.get("detektor", False),
        "verifier": vstat.get("verifier", False),
    }


def _predict(model, img_bgr: np.ndarray, classes: list) -> Dict:
    """Enhance -> predict -> probabilitas per kelas + top1."""
    enhanced = enhance(img_bgr)
    res = model.predict(enhanced, imgsz=IMGSZ, verbose=False)[0]

    # probs.data urutannya mengikuti model.names
    probs = res.probs.data.tolist()
    names = res.names  # dict {idx: nama}
    prob_map = {names[i]: float(probs[i]) for i in range(len(probs))}
    # samakan urutan & ejaan dengan daftar kelas frontend
    ordered = {c: prob_map.get(c, 0.0) for c in classes}

    top_class = max(ordered, key=ordered.get)
    top_conf = ordered[top_class]
    return {
        "kelas": top_class,
        "confidence": round(top_conf, 4),
        "prob": {k: round(v, 4) for k, v in ordered.items()},
        "diterima": top_conf >= CONFIDENCE_THRESHOLD,
    }


def _decode(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gagal membaca gambar")
    return img


def _downscale(img: np.ndarray, cap: int = 1280) -> np.ndarray:
    """Perkecil bila sisi terpanjang > cap. Gambar camera-trap sering 16MP;
    enhance/predict pada resolusi segitu boros — model sortir toh pakai 224px."""
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= cap:
        return img
    s = cap / m
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def classify_sort(image_bytes: bytes) -> Dict:
    if _sort_model is None:
        raise RuntimeError("Model sortir belum dimuat")
    img = _decode(image_bytes)
    small = _downscale(img, 1280)           # <- kunci kecepatan
    enhanced = enhance(small)               # enhance SEKALI, di gambar kecil
    res = _sort_model.predict(enhanced, imgsz=IMGSZ, verbose=False)[0]
    probs = res.probs.data.tolist()
    names = res.names
    prob_map = {names[i]: float(probs[i]) for i in range(len(probs))}
    ordered = {c: prob_map.get(c, 0.0) for c in SORT_CLASSES}
    top = max(ordered, key=ordered.get)
    out = {
        "kelas": top,
        "confidence": round(ordered[top], 4),
        "prob": {k: round(v, 4) for k, v in ordered.items()},
        "diterima": ordered[top] >= CONFIDENCE_THRESHOLD,
    }
    sc = float(blur_score(img))             # blur pada resolusi penuh (ambang tetap valid)
    out["blur"] = round(sc, 2)
    out["blur_label"] = blur_label(sc)
    out["blur_after"] = round(float(blur_score(enhanced)), 2)

    # ── VERIFIKASI TAHAP DUA ─────────────────────────────────────
    # Hanya bila hasil sortir 'kukang'. Crop objek lalu cek ulang.
    out["dikoreksi"] = False
    out["perlu_review"] = False
    if VERIFY_ENABLED and out["kelas"] == "kukang" and verifier.available():
        v = verifier.verify_is_kukang(small)  # crop + cek: benar kukang?
        out["verifikasi"] = v
        # Verifier SETUJU kukang -> tetap kukang. Selain itu (bilang bukan_kukang,
        # ragu, atau objek tak terdeteksi) -> objek tak sesuai -> perlu_review.
        if not (v["status"] == "ok" and v["is_kukang"] is True):
            out["perlu_review"] = True
    return out


def _behavior_pred(model, img_bgr: np.ndarray, imgsz: int):
    """Prediksi satu model behavior. Kembalikan (kelas_top, dict prob huruf kecil)."""
    res = model.predict(img_bgr, imgsz=imgsz, verbose=False)[0]
    probs = res.probs.data.tolist()
    names = res.names
    prob_map = {names[i].lower(): float(probs[i]) for i in range(len(probs))}
    ordered = {c: prob_map.get(c, 0.0) for c in BEHAVIOR_CLASSES}
    return max(ordered, key=ordered.get), ordered


def classify_behavior(image_bytes: bytes) -> Dict:
    """Behavior DUA-TAHAP:
      TAHAP 1 (full) : kukang_behaviour_best.pt pada gambar penuh (enhance, imgsz 320)
      TAHAP 2 (crop) : behaviour_best_crop.pt pada crop objek (imgsz 224)
    Kedua tahap SEPAKAT & cukup yakin -> masuk kelasnya.
    BEDA / ragu / hewan tak terdeteksi -> perlu_review.
    """
    if _behavior_model is None:
        raise RuntimeError("Model behavior (full) belum dimuat")
    img = _decode(image_bytes)

    # TAHAP 1 — FULL image (enhance, sama seperti training notebook full)
    topA, ordA = _behavior_pred(_behavior_model, enhance(img), FULL_IMGSZ)

    # TAHAP 2 — CROP objek (mentah)
    crop, ok = verifier.detect_and_crop(img)
    if _behavior_crop_model is not None:
        topB, ordB = _behavior_pred(_behavior_crop_model, crop if ok else img, IMGSZ)
    else:
        topB, ordB = topA, ordA

    sepakat = (topA == topB)
    confA, confB = ordA[topA], ordB[topB]
    kelas = topB if ok else topA      # crop khusus jadi penentu bila crop tersedia
    perlu_review = (not ok) or (not sepakat) \
        or (confA < BEHAVIOR_REVIEW_THRESHOLD) or (confB < BEHAVIOR_REVIEW_THRESHOLD)

    return {
        "kelas": kelas,
        "confidence": round((confA + confB) / 2, 4),
        "prob": {k: round(ordB[k], 4) for k in BEHAVIOR_CLASSES},
        "crop": ok,
        "sepakat": sepakat,
        "full": topA,      # hasil tahap 1 (full)
        "crop_pred": topB,  # hasil tahap 2 (crop)
        "perlu_review": perlu_review,
        "diterima": (not perlu_review),
    }


def enhance_image(image_bytes: bytes) -> Dict:
    """Enhance saja, tanpa model. Kembalikan JPEG hasil + skor blur sebelum/sesudah.
    Dipakai oleh alat Enhance mandiri (tab terpisah di frontend)."""
    img = _decode(image_bytes)
    before = float(blur_score(img))
    out = enhance(img)
    after = float(blur_score(out))
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("Gagal mengekode hasil enhance")
    return {
        "image": buf.tobytes(),               # bytes JPEG hasil enhance
        "blur_before": round(before, 2),
        "blur_after": round(after, 2),
        "blur_label_before": blur_label(before),
        "blur_label_after": blur_label(after),
    }
