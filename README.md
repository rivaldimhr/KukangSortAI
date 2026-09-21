# KukangSort AI — Backend (FastAPI)

Backend inference untuk frontend `kukangsort-ai.html`. Memuat dua model
YOLO11n-cls: **sortir** (kukang/unknown/delete) dan **behavior**
(bergerak/grooming/makan). Preprocessing `enhance()` disalin persis dari
`kukang_behaviour.ipynb`, jadi input model saat inference identik dengan
saat training.

## Prasyarat

- **Tanpa Docker:** Python 3.11, `venv` sudah tersedia di
  `kukangsort-backend/backend/venv` (kalau belum ada, lihat langkah manual di
  bawah).
- **Dengan Docker:** Docker Desktop terinstall (Windows/Mac) atau Docker
  Engine + Compose plugin (Linux).
- Model `.pt` di `kukangsort-backend/backend/models/` — lihat
  [README backend](kukangsort-backend/backend/README.md#download-model-wajib-sebelum-jalan--build-docker)
  untuk cara mendapatkannya kalau folder itu kosong.

## Cara Menjalankan

**Opsi A — tanpa Docker (paling cepat di laptop sendiri, Windows):**

```powershell
cd D:\kukangsort-ai-2tahap
.\start.bat
```

Otomatis menyalakan backend (port 8000), server frontend (port 5500), dan
membuka `http://localhost:5500/kukangsort-ai.html` di browser.

**Opsi B — dengan Docker (portable, bisa dicoba di komputer lain):**

```bash
docker compose up --build
```

Lalu buka `http://localhost:5500`. Backend jalan di `http://localhost:8000`.

**Opsi C — manual (kalau mau kontrol penuh / debugging):**

```bash
cd kukangsort-backend/backend
python -m venv venv
venv\Scripts\activate           # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Buka `kukangsort-ai.html` lewat server HTTP lokal (`python -m http.server 5500`
di root proyek), jangan langsung double-click file HTML — browser modern
(Chrome) memblokir fetch dari `file://` ke `localhost` karena kebijakan
Private Network Access.

## Struktur

```
backend/
├── main.py            # FastAPI: endpoint sort, behavior, batch, pack
├── inference.py       # muat model + prediksi
├── preprocessing.py   # CLAHE + unsharp + normalize (dari notebook)
├── exif_time.py       # baca EXIF DateTimeOriginal + rename per waktu
├── requirements.txt
└── models/
    ├── sortir_best.pt      # <- taruh best.pt tahap 1 di sini
    └── behaviour_best.pt   # <- best.pt dari kukang_behaviour.ipynb
```

## 1. Model sudah terpasang

Folder `models/` **sudah berisi** dua bobot asli dari demo Streamlit-mu:

- `sortir_best.pt` — kukang / unknown / delete (`model.names`: delete, kukang, unknown)
- `behaviour_best.pt` — bergerak / grooming / makan

Kode memetakan probabilitas berdasarkan **nama kelas**, bukan indeks, jadi
urutan `SORT_CLASSES` / `BEHAVIOR_CLASSES` di `inference.py` hanya mengatur
urutan tampil di frontend — tidak akan salah label meski `model.names`
berurutan lain.

Folder `sample_testing/` berisi 6 foto camera trap untuk uji cepat.

## 2. Jalankan

```bash
cd backend
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Cek: buka http://localhost:8000/health — harus tampil
`{"models": {"sortir": true, "behavior": true}}`.

Uji cepat tanpa frontend (sudah terbukti jalan dengan model asli):

```bash
curl -F "file=@sample_testing/Copy of 05210936.JPG" http://localhost:8000/api/sort
# -> {"kelas":"kukang","confidence":0.9887,"prob":{...},"blur":58.96,...}
```

## 3. Sambungkan frontend

Di `kukangsort-ai.html`, baris awal `<script>`:

```js
const API_BASE = 'http://localhost:8000';
```

- Isi = backend dipakai (model asli).
- Kosongkan `''` = mode simulasi (probabilitas acak, tanpa server).

Buka `kukangsort-ai.html` di browser. Kalau backend hidup, log di kartu
"Data mentah" menampilkan **"Backend tersambung — memakai model asli."**

> CORS sudah dibuka `*` untuk pengembangan. Untuk produksi, ganti
> `allow_origins` di `main.py` dengan domain frontend-mu.

## Endpoint

| Method | Path | Fungsi |
|---|---|---|
| GET | `/health` | status model + threshold |
| POST | `/api/sort` | 1 gambar → kelas sortir + prob + blur |
| POST | `/api/behavior` | 1 gambar → kelas behavior + prob |
| POST | `/api/enhance` | 1 gambar → JPEG hasil enhance (blur di header) |
| POST | `/api/enhance/pack` | banyak gambar → ZIP `enhanced_*` + manifest blur |
| POST | `/api/sort/batch` | banyak gambar → hasil + nama baru (EXIF di server) |
| POST | `/api/pack` | manifest + gambar → ZIP folder per kelas |

Frontend saat ini memanggil `/api/sort` dan `/api/behavior` per gambar
(paling sederhana). Untuk dataset besar, pindah ke `/api/sort/batch` dan
`/api/pack` supaya EXIF dan pengemasan ZIP dikerjakan server, bukan browser.

## Alat Enhance (tab terpisah)

Tab "Alat · Enhance" di frontend adalah alat mandiri: peneliti memasukkan
folder mana pun (hasil sorting yang sudah diunduh, folder kukang, dsb),
citra di-enhance (CLAHE → unsharp → normalize), lalu diunduh sebagai ZIP
berisi berkas `enhanced_*`.

Sengaja dipisah dari sortir. Alur sortir menghasilkan **arsip bukti** —
foto asli yang tidak dimanipulasi. Enhance menghasilkan citra yang
dipertajam untuk **dilihat mata manusia**. Memisahkannya menjaga agar
kedua hal itu tidak tercampur: sortir selalu original, enhance selalu
pilihan sadar peneliti.

## Di mana menjalankan backend?

- **Uji coba / sidang, model kecil:** laptop sendiri, CPU cukup untuk
  YOLO11n-cls (n = nano). Jalankan `uvicorn` seperti di atas.
- **Dataset besar / butuh GPU:** sewa GPU (mis. RunPod, Vast.ai, atau
  server lab kampus). Sama, `uvicorn` + buka port.
- **Colab sebagai server sementara:** bisa, bungkus dengan `pyngrok` untuk
  URL publik. Praktis untuk demo, tapi mati saat runtime reset — bukan untuk
  produksi. Kalau mau, minta skrip Colab-nya nanti.

Rekomendasi: untuk demo TA, **laptop sendiri** paling aman — tidak bergantung
koneksi, tidak ada URL yang bisa mati saat presentasi.
