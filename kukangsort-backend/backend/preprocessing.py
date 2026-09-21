"""
preprocessing.py — DISALIN PERSIS dari kukang_behaviour.ipynb (Step 3).

Preprocessing di inference WAJIB identik dengan yang dipakai saat training,
kalau tidak, distribusi input model bergeser dan akurasi anjlok tanpa jejak.
Jangan ubah nilai apa pun di sini tanpa melatih ulang model.

Pipeline: CLAHE -> Unsharp Masking -> Normalize Brightness
Blur score: Laplacian Variance (untuk info, bukan bagian preprocessing model).
"""
import cv2
import numpy as np


def blur_score(img_bgr: np.ndarray) -> float:
    """Laplacian Variance — tinggi = tajam, rendah = blur."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return round(cv2.Laplacian(gray, cv2.CV_64F).var(), 2)


def blur_label(score: float) -> str:
    if score < 50:
        return "sangat blur"
    elif score < 200:
        return "cukup blur"
    elif score < 500:
        return "sedang"
    return "tajam"


def is_grayscale(img_bgr: np.ndarray) -> bool:
    b, g, r = cv2.split(img_bgr)
    diff_bg = np.mean(np.abs(b.astype(int) - g.astype(int)))
    diff_br = np.mean(np.abs(b.astype(int) - r.astype(int)))
    return (diff_bg + diff_br) / 2 < 5


def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if is_grayscale(img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def unsharp_mask(img_bgr: np.ndarray, strength: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=2)
    return cv2.addWeighted(img_bgr, 1 + strength, blurred, -strength, 0)


def normalize_brightness(img_bgr: np.ndarray, target_mean: float = 120.0) -> np.ndarray:
    current_mean = np.mean(img_bgr)
    if current_mean < 1:
        return img_bgr
    scale = target_mean / current_mean
    return np.clip(img_bgr.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def enhance(img_bgr: np.ndarray) -> np.ndarray:
    img = apply_clahe(img_bgr)
    img = unsharp_mask(img)
    img = normalize_brightness(img)
    return img
