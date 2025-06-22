import joblib
import base64
import socket
import module
import librosa
import streamlit as st
import librosa.display
import requests.exceptions
import matplotlib.pyplot as plt

# UIiiiiiiii
st.set_page_config(page_title="Klasifikasi Sirine", layout="centered")

# BACKGROUND
def add_background(image_file, opacity=0.7):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, {opacity}), rgba(0, 0, 0, {opacity})), url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# atur background
add_background("background.jpg", opacity=0.8)

# LOAD MODEL DAN SCALER
model = joblib.load("svm_model.pkl")
scaler = joblib.load("svm_scaler.pkl")
encoder = joblib.load("svm_encoder.pkl")

label_names = ["ambulance", "firetruck", "traffic"]

# FUNGSI VISUALISASI
def plot_waveform(y, sr):
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr)
    ax.set_title("Waveform")
    plt.tight_layout()
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    return fig

def plot_mfcc(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    ax.set_title('MFCC')
    fig.colorbar(img, ax=ax)
    plt.tight_layout()
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    return fig

# HEADER
with st.container():
    st.markdown("<h1 style='text-align: center;'>Klasifikasi Suara Sirine Darurat</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Deteksi sirine atau non-sirine mobil ambulans atau pemadam melalui audio</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# INPUT AUDIO
with st.container():
    try:
        uploaded_file = st.file_uploader("Upload file audio (.wav)", type=["wav"])

        if uploaded_file:
            col1, col2 = st.columns([1, 3])

            with col2:
                st.audio(uploaded_file, format="audio/wav")

            with col1:
                sr, y = module.load_audio(uploaded_file)
                y_, sr_ = librosa.load(uploaded_file)
                durasi = module.get_duration(signal=y, sr=sr)
                st.write(f"Durasi: {durasi:.2f} detik")
                st.write(f"Sample Rate: {sr} Hz")

            st.markdown("---")

            # KLASIFIKASI
            if st.button("Mulai Klasifikasi"):
                try:
                    with st.spinner("Memproses audio..."):
                        fitur = module.extract_all_features(sr=sr, signal=y)
                        uniform_fitur = module.uniform_data(fitur)
                        fitur_scaled = scaler.transform(uniform_fitur)
                        hasil = model.predict(fitur_scaled)
                        hasil_decoded = encoder.inverse_transform(hasil)
                        confidence = model.predict_proba(fitur_scaled)[0]

                    st.success(f"Hasil Klasifikasi: **{hasil_decoded[0]}**")
                    st.info(f"Keyakinan model: **{confidence[hasil[0]] * 100:.2f}%**")

                    # VISUALISASI
                    st.markdown("<h3 style='text-align: center;'>Visualisasi Audio</h3>", unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(plot_waveform(y, sr))
                    with col2:
                        st.pyplot(plot_mfcc(y_, sr_))

                except (requests.exceptions.ConnectionError, socket.error, ConnectionError, TimeoutError):
                    st.error("Terjadi kesalahan saat proses. Silakan coba ulang.")

    except (requests.exceptions.ConnectionError, socket.error, ConnectionError, TimeoutError):
        st.error("Terjadi kesalahan jaringan. Silakan coba ulang.")

# FOOTER
with st.container():
    st.markdown("---")
    st.markdown("<p style='text-align: center; font-size: 14px;'>Dikembangkan oleh Kelompok 2 · Streamlit + Librosa + SVM</p>", unsafe_allow_html=True)
