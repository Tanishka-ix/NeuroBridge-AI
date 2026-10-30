import streamlit as st
import librosa
import numpy as np
import tempfile
import os

st.title("NeuroBridge AI")
st.write("Baby Cry Audio Analysis")

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "ogg", "mp3"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp:
        temp.write(uploaded_file.getbuffer())
        audio_file = temp.name

    audio, sample_rate = librosa.load(audio_file, sr=None)

    duration = librosa.get_duration(y=audio, sr=sample_rate)

    rms = librosa.feature.rms(y=audio)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sample_rate
    )
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sample_rate, n_mfcc=13
    )

    st.success("Audio analyzed successfully!")

    st.subheader("Audio Features")

    st.write(f"**Duration:** {duration:.2f} seconds")
    st.write(f"**Sample Rate:** {sample_rate} Hz")
    st.write(f"**Average Energy:** {np.mean(rms):.4f}")
    st.write(f"**Zero-Crossing Rate:** {np.mean(zero_crossing_rate):.4f}")
    st.write(f"**Spectral Centroid:** {np.mean(spectral_centroid):.2f} Hz")
    st.write(f"**Average MFCC:** {np.mean(mfcc):.4f}")

    os.remove(audio_file)