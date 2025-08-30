# app.py
import json, soundfile as sf, numpy as np, librosa, streamlit as st
from src.align_word import transcribe_with_words
from src.score import score_sentence
from src.tips import tips_for

st.set_page_config(page_title="Sprich! 🇩🇪", page_icon="🇩🇪", layout="centered")

prompts = json.load(open("data/prompts_de.json","r",encoding="utf-8"))
st.title("Sprich! – German Pronunciation Coach")
choice = st.selectbox("Choose a sentence to practice:", options=list(enumerate(prompts)), format_func=lambda x: x[1])
idx, text = choice

st.audio(f"data/ref_audio/{idx:03d}.wav", format="audio/wav")
user_file = st.file_uploader("Record and upload your attempt (WAV/MP3)", type=["wav","mp3"])

if user_file and st.button("Score my pronunciation"):
    # load audio
    y_ref, sr_ref = librosa.load(f"data/ref_audio/{idx:03d}.wav", sr=16000, mono=True)
    y_usr, sr_usr = librosa.load(user_file, sr=16000, mono=True)

    # get word timestamps
    ref_words = transcribe_with_words(f"data/ref_audio/{idx:03d}.wav", language="de")
    usr_words = transcribe_with_words(user_file, language="de")
    text_words = text.replace("!", "").replace("?", "").replace(".", "").split()

    # score
    score, per_word, meta = score_sentence(y_ref, y_usr, 16000, ref_words, usr_words, text_words)
    st.metric("Overall score", f"{score:.0f} / 100")
    st.write("Per-word feedback (darker = better):")
    for w in per_word:
        shade = int(255 - 155*w["score"])
        st.markdown(f"<span style='background-color: rgb(255,{shade},{shade}); padding:4px 8px; border-radius:8px;'>{w['word']} ({w['score']*100:.0f}%)</span> ", unsafe_allow_html=True)
    st.caption(f"Prosody: {meta['prosody']:.2f} · Fluency: {meta['fluency']:.2f}")
    for tip in tips_for(per_word):
        st.info(tip)
