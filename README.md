# Oil Palm Ripeness Classifier — Web App

Web app Streamlit untuk model penelitian:
**"Field Validation and Energy-Aware Edge Deployment of Hybrid GA-PSO-SA
Multi-Objective CNN with Self-Adaptive Dynamic Attention for Oil Palm
Ripeness Classification"**

## Isi folder

```
oil_palm_app/
├── app.py                  # Aplikasi web (Streamlit)
├── model.py                # Definisi arsitektur model (ResNet18 + custom head)
├── requirements.txt        # Daftar dependency
├── model_weights/
│   └── best_model_multiobjective.pth   # Bobot model hasil training
├── assets/
│   ├── logo_uny.png
│   └── logo_bima.png
└── README.md
```

## ⚠️ Penting — validasi arsitektur

File `.pth` yang Anda unggah **hanya berisi state_dict (bobot)**, tanpa kode
kelas model. Arsitektur di `model.py` sudah saya rekonstruksi berdasarkan
key-key yang ada di dalam checkpoint, dan **berhasil dimuat 100% cocok**
(`strict=True`, tanpa missing/unexpected keys):

- Backbone: ResNet-18 standar (torchvision)
- Head kustom: Dropout → Linear(512,512) → ReLU → BatchNorm1d → Dropout → Linear(512,4)

Jika model asli Anda punya modul **Self-Adaptive Dynamic Attention** dengan
parameter/layer tambahan yang terlatih terpisah, namun tidak muncul di
checkpoint ini, kemungkinan:
1. Checkpoint yang diunggah adalah versi backbone saja (tanpa attention module tersimpan terpisah), atau
2. Attention module Anda tidak punya learnable parameter (mis. hanya operasi non-parametrik).

**Rekomendasi:** jika hasil prediksi terasa kurang akurat dibanding hasil
training/testing Anda, kirimkan kode definisi model (class `nn.Module`)
yang dipakai saat training, agar `model.py` bisa disesuaikan 100% persis.

Urutan kelas yang dipakai (index 0–3): **Unripe, Underripe, Ripe, Overripe**
— sesuaikan `CLASS_NAMES` di `model.py` jika urutan asli berbeda.

## Menjalankan secara lokal

```bash
# 1. Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependency
pip install -r requirements.txt

# 3. Jalankan
streamlit run app.py
```

App akan terbuka otomatis di browser pada `http://localhost:8501`.

## Deploy online (gratis) — Streamlit Community Cloud

Langkah lengkap dari nol sampai app online dengan URL publik:

1. **Buat akun GitHub** (kalau belum punya) di https://github.com
2. **Buat repository baru** (public), misal nama `oil-palm-ripeness-app`
3. **Upload semua isi folder `oil_palm_app/`** ke repo tersebut:
   - Cara termudah (tanpa command line): buka repo di GitHub → tombol
     **"Add file" → "Upload files"** → drag & drop semua file/folder
     (`app.py`, `model.py`, `requirements.txt`, folder `model_weights/`,
     folder `assets/`, `README.md`) → klik **Commit changes**
   - Atau via command line:
     ```bash
     cd oil_palm_app
     git init
     git add .
     git commit -m "Initial commit"
     git branch -M main
     git remote add origin https://github.com/USERNAME/oil-palm-ripeness-app.git
     git push -u origin main
     ```
4. **Buka https://share.streamlit.io** → Sign in pakai akun GitHub yang sama
5. Klik **"New app"**
6. Pilih repository yang baru dibuat, branch `main`, dan **Main file path**
   isi dengan `app.py`
7. Klik **Deploy**. Tunggu 2–5 menit (Streamlit Cloud akan install semua
   dependency dari `requirements.txt` otomatis)
8. Setelah selesai, app akan online dengan URL publik seperti:
   `https://oil-palm-ripeness-app-xxxxx.streamlit.app`
9. URL ini bisa dibagikan ke siapa saja untuk mengakses app tanpa perlu
   install apa pun di sisi mereka.

**Catatan:** file `.pth` berukuran ~45 MB, masih dalam batas wajar untuk
GitHub (limit 100 MB per file) dan Streamlit Community Cloud (limit ~1 GB
per repo untuk paket gratis).

**Update app setelah online:** setiap kali Anda push perubahan baru ke
branch `main` di GitHub, Streamlit Cloud otomatis redeploy app dalam
beberapa menit — tidak perlu setup ulang.

## Deploy alternatif

- **Hugging Face Spaces** (gratis, mendukung Streamlit/Gradio langsung dari repo)
- **Railway** / **Render** — cocok kalau ingin backend terpisah (FastAPI) + frontend sendiri
- **Docker + VPS** — kontrol penuh, cocok untuk deployment produksi/edge server

## Tentang aspek "Energy-Aware Edge Deployment"

App ini menampilkan **waktu inferensi (ms)** setiap prediksi sebagai indikator
efisiensi komputasi. Namun ini BUKAN pengukuran konsumsi daya (Watt) yang
sesungguhnya. Untuk validasi energy-aware yang valid secara penelitian, lakukan
pengukuran daya langsung di perangkat edge target (mis. Jetson Nano/Xavier,
Raspberry Pi) menggunakan tool seperti `tegrastats`/`jtop` (Jetson) atau
`powertop` (Linux umum), dan bandingkan dengan baseline model non-optimized.

## Menyesuaikan aplikasi

- **Ganti label kelas**: edit `CLASS_NAMES` di `model.py`
- **Ganti ukuran input**: edit `IMG_SIZE` di `model.py` (default 224x224)
- **Ganti normalisasi**: edit `IMAGENET_MEAN` / `IMAGENET_STD` di `model.py`
  jika training menggunakan normalisasi berbeda (mis. min-max 0–1 saja)
- **Tambah info metadata gambar (GPS, timestamp)** untuk keperluan "field
  validation": bisa ditambahkan melalui EXIF data dari `Pillow`
