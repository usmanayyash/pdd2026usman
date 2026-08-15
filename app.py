"""
Web App: Field Validation and Energy-Aware Edge Deployment of
Hybrid GA-PSO-SA Multi-Objective CNN
Oil Palm Ripeness Classification

Kegiatan: Penelitian Disertasi Doktor (PDD) Tahun 2026
Peneliti: Fatchul Arifin, Rustam Asnawai, Usman

Jalankan lokal:
    pip install -r requirements.txt
    streamlit run app.py
"""

import base64
import io
import os
import time

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import load_model, CLASS_NAMES, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

# ------------------------------------------------------------------
# Konfigurasi halaman
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Oil Palm Ripeness Classifier | PDD 2026",
    page_icon="🌴",
    layout="centered",
)

BASE_DIR = os.path.dirname(__file__)
WEIGHTS_PATH = os.path.join(BASE_DIR, "model_weights", "best_model_multiobjective.pth")
LOGO_UNY_PATH = os.path.join(BASE_DIR, "assets", "logo_uny.png")
LOGO_BIMA_PATH = os.path.join(BASE_DIR, "assets", "logo_bima.png")

RESEARCH_TITLE = (
    "Field Validation and Energy-Aware Edge Deployment of "
    "Hybrid GA-PSO-SA Multi-Objective CNN"
)
ACTIVITY_NAME = "Penelitian Disertasi Doktor (PDD) Tahun 2026"
RESEARCHERS = ["Fatchul Arifin", "Rustam Asnawai", "Usman"]

CLASS_COLOR = {
    "Overripe": "#795548",
    "Ripe": "#FF9800",
    "Underripe": "#CDDC39",
    "Unripe": "#8BC34A",
}

CLASS_DESC = {
    "Unripe": "Buah belum matang — kandungan minyak masih rendah, disarankan belum dipanen.",
    "Underripe": "Buah mendekati matang — bisa dipanen dalam waktu dekat.",
    "Ripe": "Buah matang optimal — waktu terbaik untuk dipanen (kandungan minyak maksimal).",
    "Overripe": "Buah terlalu matang — kualitas minyak mulai menurun, segera panen/olah.",
}


# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        .header-card {
            background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 55%, #558B2F 100%);
            border-radius: 16px;
            padding: 28px 24px;
            margin-bottom: 22px;
            box-shadow: 0 8px 24px rgba(27, 94, 32, 0.25);
        }
        .header-title {
            color: #FFFFFF;
            font-size: 22px;
            font-weight: 700;
            line-height: 1.35;
            margin-bottom: 14px;
        }
        .header-researchers {
            color: #F1F8E9;
            font-size: 14px;
            font-weight: 400;
            border-top: 1px solid rgba(255,255,255,0.25);
            padding-top: 10px;
        }
        .header-researchers b {
            color: #FFFFFF;
        }
        .result-card {
            border-radius: 14px;
            padding: 18px 20px;
            background: #FFFFFF;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            border: 1px solid #eee;
        }
        .badge {
            display: inline-block;
            color: white;
            padding: 5px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            margin-top: 6px;
        }
        .footer-note {
            font-size: 12px;
            color: #777;
            text-align: center;
            margin-top: 30px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def img_to_b64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def render_header():
    uny_b64 = img_to_b64(LOGO_UNY_PATH)
    bima_b64 = img_to_b64(LOGO_BIMA_PATH)

    logo_col1, title_col, logo_col2 = st.columns([1, 4, 1])
    with logo_col1:
        if uny_b64:
            st.markdown(
                f'<img src="data:image/png;base64,{uny_b64}" style="width:100%;max-width:80px;">',
                unsafe_allow_html=True,
            )
    with logo_col2:
        if bima_b64:
            st.markdown(
                f'<img src="data:image/png;base64,{bima_b64}" style="width:100%;max-width:80px;">',
                unsafe_allow_html=True,
            )
    with title_col:
        st.markdown(
            f"""
            <div style="text-align:center; padding-top:4px;">
                <div style="font-size:12px; font-weight:700; letter-spacing:1.5px;
                            text-transform:uppercase; color:#2E7D32;">
                    {ACTIVITY_NAME}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    researchers_str = ", ".join(RESEARCHERS)
    st.markdown(
        f"""
        <div class="header-card">
            <div class="header-title">🌴 {RESEARCH_TITLE}</div>
            <div class="header-researchers">Peneliti: <b>{researchers_str}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Load model (cache supaya tidak reload tiap interaksi)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(WEIGHTS_PATH, device=device)
    return model, device


def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(image).unsqueeze(0)


def predict(model, device, image: Image.Image):
    tensor = preprocess_image(image).to(device)

    # Ukur waktu inferensi -> relevan untuk aspek "energy-aware edge deployment"
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    probs = probs.cpu().numpy()
    pred_idx = int(probs.argmax())
    return pred_idx, probs, elapsed_ms


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
inject_css()
render_header()

model, device = get_model()
st.caption(f"⚙️ Perangkat inferensi: `{device.upper()}`")

st.divider()

uploaded_file = st.file_uploader(
    "📤 Unggah gambar buah kelapa sawit (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
)

col_input, col_result = st.columns(2)

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

    with col_input:
        st.image(image, caption="Gambar diunggah", use_container_width=True)

    pred_idx, probs, elapsed_ms = predict(model, device, image)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100

    with col_result:
        st.markdown(
            f"""
            <div class="result-card">
                <div style="font-size:14px; color:#666;">Hasil Klasifikasi</div>
                <div style="font-size:24px; font-weight:700; color:#1B5E20; margin:4px 0;">
                    {pred_class}
                </div>
                <span class="badge" style="background-color:{CLASS_COLOR[pred_class]};">
                    Keyakinan: {confidence:.1f}%
                </span>
                <p style="margin-top:12px; color:#444; font-size:14px;">{CLASS_DESC[pred_class]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("⏱️ Waktu inferensi", f"{elapsed_ms:.1f} ms")

    st.divider()
    st.subheader("📊 Distribusi probabilitas semua kelas")
    for i, cname in enumerate(CLASS_NAMES):
        st.write(f"**{cname}** (index {i})")
        st.progress(float(probs[i]), text=f"{probs[i]*100:.1f}%")

else:
    st.info("Silakan unggah foto buah kelapa sawit untuk mulai klasifikasi.")

with st.expander("ℹ️ Tentang model ini"):
    st.write(
        "Model ini adalah CNN berbasis **ResNet-18** yang dioptimasi menggunakan "
        "pendekatan hybrid **Genetic Algorithm (GA)**, **Particle Swarm Optimization (PSO)**, "
        "dan **Simulated Annealing (SA)** dengan skema multi-objective, untuk klasifikasi "
        "tingkat kematangan buah kelapa sawit ke dalam 4 kelas: "
        f"**{', '.join(CLASS_NAMES)}**."
    )

st.divider()
st.markdown(
    """
    <div class="footer-note">
        Catatan energi & edge deployment: waktu inferensi di atas diukur langsung pada
        perangkat yang menjalankan app ini (CPU/GPU server), bukan hasil pengukuran daya
        riil pada perangkat edge (mis. Jetson/Raspberry Pi). Untuk validasi energy-aware
        yang sesungguhnya, ukur konsumsi daya (Watt) pada perangkat target menggunakan
        power meter atau tool seperti <code>jtop</code> (Jetson) / <code>powertop</code>.
    </div>
    """,
    unsafe_allow_html=True,
)
