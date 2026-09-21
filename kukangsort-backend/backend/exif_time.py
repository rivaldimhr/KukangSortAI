"""
exif_time.py — sumber waktu bidikan, urut prioritas:
  1. EXIF DateTimeOriginal / DateTimeDigitized  (paling sahih)
  2. Pola MMDD pada nama berkas                 (cadangan, tanpa jam/tahun)
  3. Gagal -> None                              (citra ditandai, tak ikut filter waktu)

Ini menggantikan file.lastModified di frontend, yang tidak andal kalau
berkas sudah pernah disalin (mis. nama 'Copy_of_...').
"""
import re
from datetime import datetime
from typing import Optional, Tuple

from PIL import Image
from PIL.ExifTags import TAGS

_MMDD = re.compile(r"(\d{2})(\d{2})(\d{4})")  # 05230529 -> bulan 05, tgl 23


def _from_exif(path: str) -> Optional[datetime]:
    try:
        img = Image.open(path)
        exif = img._getexif()
        if not exif:
            return None
        tagmap = {TAGS.get(k, k): v for k, v in exif.items()}
        for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
            val = tagmap.get(key)
            if val:
                # format EXIF standar: "YYYY:MM:DD HH:MM:SS"
                return datetime.strptime(str(val), "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None
    return None


def _from_name(name: str) -> Optional[datetime]:
    m = _MMDD.search(name)
    if not m:
        return None
    bulan, tgl = int(m.group(1)), int(m.group(2))
    if 1 <= bulan <= 12 and 1 <= tgl <= 31:
        return datetime(datetime.now().year, bulan, tgl)
    return None


def read_taken(path: str, name: str) -> Tuple[Optional[datetime], str]:
    """Kembalikan (datetime | None, sumber)."""
    d = _from_exif(path)
    if d:
        return d, "exif"
    d = _from_name(name)
    if d:
        return d, "nama berkas"
    return None, "tidak terbaca"


def rename_by_time(taken: Optional[datetime], orig: str) -> str:
    """YYYYMMDD_HHMMSS__<nama_asli>, atau TANPA-WAKTU__ kalau gagal."""
    stem, dot, ext = orig.rpartition(".")
    stem = stem or orig
    ext = ("." + ext).lower() if dot else ".jpg"
    safe = re.sub(r'[\\/:*?"<>|]', "_", stem)
    if taken:
        return taken.strftime("%Y%m%d_%H%M%S") + "__" + safe + ext
    return "TANPA-WAKTU__" + safe + ext
