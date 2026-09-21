# Pembaruan: kecepatan + behavior crop-based

## Apa yang berubah
1. **Behavior kini crop-based** (deteksi -> crop -> klasifikasi), pakai model
   3-kelas baru (Bergerak/Makan/grooming, ~95%). File: models/behaviour_best.pt.
   - Behavior dijalankan pada CROP mentah (tanpa enhance), konsisten dgn training.
2. **Detektor diganti yolo11n @ 416** (dari yolo11x @ 640) di verifier.py.
   - ~33x lebih ringan di CPU; pada uji tetap menemukan hewan dgn baik.
   - models/yolo11x.pt DIHAPUS (tak terpakai). Bisa dikembalikan bila mau teliti:
     ubah DETECTOR_WEIGHTS & DETECT_IMGSZ di verifier.py, taruh lagi yolo11x.pt.
3. **Sortir dipercepat**: gambar diperkecil ke 1280px sebelum enhance/predict.
   - enhance() tadinya jalan di 16MP dua kali (~690ms x2). Sekarang ~30ms.
   - Hasil kelas sortir TIDAK berubah (model tetap lihat 224px).

## Dampak kecepatan (diukur)
| tahap | sebelum | sesudah |
|---|---|---|
| sortir / gambar | ~3300 ms | ~165 ms |
| behavior / gambar (crop) | (yolo11x) lambat | ~105 ms |
| **800 gambar (50% kukang)** | **~30 menit** | **~3-4 menit** |

## Alur akhir per gambar
1. Sortir (gambar diperkecil, enhance, model) -> kukang/unknown/delete.
2. Bila 'kukang': deteksi+crop -> verifikasi (kukang/bukan) -> koreksi bila perlu.
3. Tahap behavior (tab 3): untuk citra kukang -> deteksi+crop -> klasifikasi
   behavior pada crop.

## Model di models/
- sortir_best.pt      (sortir 3 kelas) — tetap
- behaviour_best.pt   (behavior 3 kelas, CROP-BASED) — BARU
- verifier_best.pt    (kukang/bukan_kukang) — tetap
- yolo11n.pt          (detektor cepat) — BARU

## Update: sortir & behavior DUA-TAHAP + folder need_review
- SORTIR: full sortir -> verifikasi crop (verifier_best.pt). Kukang yang TIDAK
  lolos verifikasi (verifier bilang bukan/ragu/tak terdeteksi) -> ditandai
  perlu_review. Output unduhan: kukang, unknown, delete, **need_review**.
- BEHAVIOR: dua model crop saling cek (behaviour_best.pt + behaviour_best_crop.pt).
  Kalau kedua model SEPAKAT & cukup yakin -> masuk kelasnya; kalau BEDA/ragu/
  tak terdeteksi -> perlu_review. Output: bergerak, makan, grooming, **need_review**.
- Model baru di models/: behaviour_best_crop.pt (model B). Ambang di inference.py:
  BEHAVIOR_REVIEW_THRESHOLD.
- Catatan: kedua model behavior sama-sama CROP-based. Kalau nanti punya model
  behavior FULL-image, ia bisa dipasang sebagai tahap-1 penuh.


## Update: behavior TAHAP-1 FULL + TAHAP-2 CROP (model final)
- TAHAP 1 (full)  : kukang_behaviour_best.pt pada gambar penuh (enhance, imgsz 320).
- TAHAP 2 (crop)  : behaviour_best_crop.pt pada crop objek (imgsz 224).
- Sepakat & yakin -> masuk kelas; beda/ragu/tak terdeteksi -> need_review.
- behaviour_best.pt LAMA tidak dipakai lagi (dihapus dari models/).
- models/ sekarang: sortir_best, kukang_behaviour_best (full), behaviour_best_crop,
  verifier_best, yolo11n.
