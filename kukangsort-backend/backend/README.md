# KukangSort AI — Backend (FastAPI) + Verifikasi Crop

Backend inference untuk `kukangsort-ai.html`. Sekarang memuat tiga model YOLO
+ satu detektor:

- `sortir_best.pt`     — sortir 3 kelas: kukang / unknown / delete  (TETAP, inti sistem)
- `behaviour_best.pt`  — behavior: bergerak / grooming / makan       (TETAP, tidak diubah)
- `verifier_best.pt`   — verifikasi 2 kelas: kukang / bukan_kukang   (BARU, pemindaian kedua)
- `yolo11x.pt`         — detektor COCO untuk crop objek              (BARU)

## Apa yang baru: pemindaian kedua (tahap verifikasi)

Masalah: model sortir kadang meloloskan objek bukan-kukang (mis. tikus) ke
kelas `kukang`. Solusinya, HANYA untuk gambar yang disortir 'kukang', backend
melakukan pengecekan kedua:

    gambar full -> yolo11x (cari & crop objek) -> verifier_best.pt (cek crop)
                                                       |
                                         kukang / bukan_kukang

- Verifier dilatih pada CROP (group-aware, precision 99%). Karena itu objek
  di-crop dulu, baru dicek — konsisten dengan cara model dilatih.
- Kebijakan aman: kukang hanya diturunkan ke `unknown` bila verifier SANGAT
  yakin bukan-kukang (confidence >= VERIFY_DEMOTE_THRESHOLD, default 0.85).
  Ini melindungi kukang asli dari crop detektor yang kadang meleset.
- Kalau verifier ragu atau objek tak terdeteksi -> tetap `kukang`, tapi diberi
  tanda `perlu_review` untuk ditinjau manusia.
- Field baru pada hasil sortir: `verifikasi`, `dikoreksi`, `kelas_awal`,
  `perlu_review`. Field lama (`waktu`, `nama_baru`, dll.) tetap seperti semula.

Model sortir & behavior TIDAK diganti. Verifier hanya menempel di atas.
Untuk mematikan sementara: set `VERIFY_ENABLED = False` di `inference.py`.

## Struktur

    backend/
    |-- main.py            # FastAPI: sort, behavior, batch, pack, enhance, /health
    |-- inference.py       # muat model + prediksi + VERIFIKASI tahap dua
    |-- verifier.py        # detektor + verifier crop (BARU)
    |-- preprocessing.py   # CLAHE + unsharp + normalize
    |-- exif_time.py       # baca EXIF + rename per waktu
    |-- requirements.txt
    `-- models/
        |-- sortir_best.pt
        |-- behaviour_best.pt
        |-- verifier_best.pt   # best.pt group-aware
        `-- yolo11x.pt         # detektor

## Model

File `*.pt` di `models/` ikut dikomit ke repo (total ~26MB, di bawah limit
GitHub) supaya begitu clone langsung bisa `uvicorn` / `docker compose up --build`
tanpa langkah tambahan.

## Menjalankan

    # (opsional, Windows/Anaconda) cegah crash OpenMP:
    #   PowerShell:  $env:KMP_DUPLICATE_LIB_OK="TRUE"
    cd kukangsort-backend/backend
    python -m venv venv
    venv\Scripts\activate            # Windows  (Linux/Mac: source venv/bin/activate)
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Lalu buka `kukangsort-ai.html`.

## Cek verifier aktif

Buka http://localhost:8000/health . Kalau muncul:
    "models": { "sortir": true, "behavior": true, "detektor": true, "verifier": true }
berarti pemindaian kedua sudah aktif.

## Menyetel

Di `inference.py`:
- `VERIFY_ENABLED` — on/off pemindaian kedua.
- `VERIFY_DEMOTE_THRESHOLD` — makin tinggi = makin hati-hati menurunkan kukang.

Untuk menangkap lebih banyak kebocoran: perbanyak crop tikus/objek lain ke
kelas `bukan_kukang` lalu latih ulang verifier (`train_verifier_grouped.py`).
