"""
main.py — FastAPI backend KukangSort AI.

Endpoint:
  GET  /health                 status model
  POST /api/sort               1 gambar  -> kelas sortir + prob + blur
  POST /api/behavior           1 gambar  -> kelas behavior + prob
  POST /api/sort/batch         banyak gambar -> daftar hasil + nama baru
  POST /api/pack               metadata JSON + gambar -> ZIP folder per kelas

Alur sesuai frontend:
  1. Sortir seluruh data raw -> folder kukang/unknown/delete
  2. Filter waktu (dilakukan di frontend memakai field 'waktu' hasil sortir)
  3. Behavior HANYA untuk kelas kukang -> folder bergerak/grooming/makan

Jalankan:  uvicorn main:app --reload --port 8000
"""
from fastapi import Response
import io
import json
import zipfile
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import inference
from exif_time import read_taken, rename_by_time

app = FastAPI(title="KukangSort AI", version="1.0")

# Frontend statis boleh dari mana saja saat pengembangan.
# Untuk produksi, ganti "*" dengan domain frontend-mu.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    status = inference.load_models()
    print("Model dimuat:", status)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "sortir": inference._sort_model is not None,
            "behavior": inference._behavior_model is not None,
            "behavior_crop": inference._behavior_crop_model is not None,
            "detektor": inference.verifier._detector is not None,
            "verifier": inference.verifier._verifier is not None,
        },
        "threshold": inference.CONFIDENCE_THRESHOLD,
    }


# ── Enhance mandiri (alat terpisah) ───────────────────────────────
# Tidak menyentuh model. Peneliti masukkan folder mana pun -> enhance
# -> unduh. Dipisah dari sortir supaya arsip bukti (original) dan
# citra yang dipertajam tidak tercampur dalam satu langkah.


@app.post("/api/enhance")
async def enhance_one(file: UploadFile = File(...)):
    """Satu gambar -> JPEG hasil enhance. Skor blur ada di header."""
    data = await file.read()
    try:
        out = inference.enhance_image(data)
    except Exception as e:
        raise HTTPException(400, str(e))
    return Response(
        content=out["image"],
        media_type="image/jpeg",
        headers={
            "X-Blur-Before": str(out["blur_before"]),
            "X-Blur-After": str(out["blur_after"]),
            "Access-Control-Expose-Headers": "X-Blur-Before, X-Blur-After",
        },
    )


@app.post("/api/enhance/pack")
async def enhance_pack(
    files: List[UploadFile] = File(...),
    root_name: str = Form("hasil_enhance"),
):
    """Banyak gambar -> ZIP berisi semua versi enhance + manifest blur.
    Nama berkas dipertahankan; hasil enhance diberi awalan 'enhanced_'."""
    buf = io.BytesIO()
    rows = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            data = await f.read()
            try:
                out = inference.enhance_image(data)
            except Exception as e:
                rows.append({"nama": f.filename, "status": "gagal: " + str(e)})
                continue
            arc = f"{root_name}/enhanced_{f.filename}"
            z.writestr(arc, out["image"])
            rows.append({
                "nama": f.filename,
                "blur_sebelum": out["blur_before"],
                "blur_sesudah": out["blur_after"],
                "status": "ok",
            })
        if rows:
            keys = ["nama", "blur_sebelum", "blur_sesudah", "status"]
            csv = ",".join(keys) + "\n"
            for r in rows:
                csv += ",".join(str(r.get(k, "")) for k in keys) + "\n"
            z.writestr(f"{root_name}/manifest_enhance.csv", csv)
    buf.seek(0)
    fname = f"{root_name}_{datetime.now():%Y%m%d%H%M}.zip"
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={fname}"},
    )


# ── Sortir satu gambar ────────────────────────────────────────────
@app.post("/api/sort")
async def sort_one(file: UploadFile = File(...)):
    data = await file.read()
    try:
        out = inference.classify_sort(data)
    except Exception as e:
        raise HTTPException(400, str(e))
    # baca waktu bidikan dari EXIF (Pillow, andal untuk Bushnell)
    taken, src = _read_taken_from_bytes(data, file.filename)
    out["waktu"] = taken.isoformat() if taken else None
    out["sumber_waktu"] = src
    out["nama_baru"] = rename_by_time(taken, file.filename)
    return out


# ── Behavior satu gambar ──────────────────────────────────────────
@app.post("/api/behavior")
async def behavior_one(file: UploadFile = File(...)):
    data = await file.read()
    try:
        out = inference.classify_behavior(data)
    except Exception as e:
        raise HTTPException(400, str(e))
    return out


# ── Sortir batch ──────────────────────────────────────────────────
# Menerima banyak gambar sekaligus. Waktu dibaca dari EXIF di sisi
# server (lebih andal daripada di browser). Mengembalikan daftar
# hasil + nama baru; frontend memakainya untuk mengisi state & filter.
@app.post("/api/sort/batch")
async def sort_batch(files: List[UploadFile] = File(...)):
    hasil = []
    for f in files:
        data = await f.read()
        # tulis sementara untuk baca EXIF (PIL butuh path atau file-like)
        taken, src = _read_taken_from_bytes(data, f.filename)
        try:
            out = inference.classify_sort(data)
        except Exception as e:
            hasil.append({"nama_asli": f.filename, "error": str(e)})
            continue
        out["nama_asli"] = f.filename
        out["nama_baru"] = rename_by_time(taken, f.filename)
        out["waktu"] = taken.isoformat() if taken else None
        out["sumber_waktu"] = src
        hasil.append(out)
    return {"jumlah": len(hasil), "hasil": hasil}


def _read_taken_from_bytes(data: bytes, name: str):
    """read_taken butuh path; pakai file sementara di memori via PIL langsung."""
    import tempfile
    import os
    from exif_time import _from_name
    try:
        with tempfile.NamedTemporaryFile(suffix="_" + name, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        taken, src = read_taken(tmp_path, name)
        os.unlink(tmp_path)
        return taken, src
    except Exception:
        d = _from_name(name)
        return (d, "nama berkas") if d else (None, "tidak terbaca")


# ── Kemas ZIP per kelas ───────────────────────────────────────────
# Frontend mengirim daftar {nama_baru, kelas/perilaku} + gambar mentah.
# Server mengelompokkan ke subfolder dan menyertakan manifest.csv.
# Ini memindahkan pengemasan berat dari browser ke server.
@app.post("/api/pack")
async def pack(
    # JSON: [{nama_baru, folder, nama_asli, ...}]
    manifest: str = Form(...),
    root_name: str = Form("hasil"),
    files: List[UploadFile] = File(...),
):
    try:
        meta = json.loads(manifest)
    except json.JSONDecodeError:
        raise HTTPException(400, "manifest bukan JSON valid")

    # petakan nama_baru -> metadata
    by_name = {m["nama_baru"]: m for m in meta}

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        rows = []
        for f in files:
            data = await f.read()
            m = by_name.get(f.filename, {})
            folder = m.get("folder", "lainnya")
            arc = f"{root_name}/{folder}/{f.filename}"
            z.writestr(arc, data)
            rows.append(m)
        # manifest.csv
       # manifest.csv — kolom diurutkan agar konsisten dan mudah dibaca
        if rows:
            urut = ["nama_baru", "nama_asli", "folder", "kelas", "perilaku",
                    "probabilitas", "waktu", "tanggal", "jam", "sumber_waktu", "path_asal", "rentang"]
            adakunci = {k for r in rows for k in r.keys()}
            keys = [k for k in urut if k in adakunci] + \
                sorted(adakunci - set(urut))
            csv = ",".join(keys) + "\n"
            for r in rows:
                csv += ",".join(f'"{r.get(k, "")}"' for k in keys) + "\n"
            # BOM UTF-8 untuk Excel
            z.writestr(f"{root_name}/manifest.csv", "\ufeff" + csv)

    buf.seek(0)
    fname = f"{root_name}_{datetime.now():%Y%m%d%H%M}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={fname}"},
    )
