# app.py
import json
import os
import io
import tempfile
from pathlib import Path

import numpy as np
import librosa
import streamlit as st

# Local modules
from src.score import score_sentence
from src.tips import tips_for
import soundfile as sf

from src.align_word import transcribe_with_words, transcribe_text

# --- Page config MUST be the first Streamlit call ---
st.set_page_config(page_title="Sprich!", page_icon="🇩🇪", layout="centered")

# --- Theme values from config.toml (optional) ---
THEME_PRIMARY = st.get_option("theme.primaryColor") or "#58CC02"
THEME_BG = st.get_option("theme.backgroundColor") or "#FFFFFF"
THEME_BG2 = st.get_option("theme.secondaryBackgroundColor") or "#F2FAEB"
THEME_TEXT = st.get_option("theme.textColor") or "#1C1E21"

# --- Global CSS (use data-testid selectors, not random emotion classes) ---
st.markdown(f"""
<style>
/* App container background (gradient) */
[data-testid="stAppViewContainer"] {{
  background: radial-gradient(1200px 600px at 20% 0%, {THEME_BG2} 0%, {THEME_BG} 40%, {THEME_BG} 100%);
}}
/* Make the top header transparent so the gradient shows through */
[data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
/* Page width + nicer metrics */
.block-container {{ max-width: 900px; }}
[data-testid="stMetricValue"] {{ font-weight: 700; }}
/* Primary buttons a touch bolder */
.stButton > button {{
  border-radius: 12px;
  padding: 0.6rem 1.1rem;
  font-weight: 600;
}}
/* Chips */
.chip {{
  color: white;
  padding: 6px 12px;
  margin: 4px;
  border-radius: 999px;
  display: inline-block;
  font-weight: 600;
}}
</style>
""", unsafe_allow_html=True)

# Custom background gradient
st.markdown("""
<style>
body {
  background: radial-gradient(1200px 600px at 20% 0%, #F2FAEB 0%, #FFFFFF 40%, #FFFFFF 100%);
}
</style>
""", unsafe_allow_html=True)


DATA_DIR = Path("data")
REF_DIR = DATA_DIR / "ref_audio"
PROMPTS_PATH = DATA_DIR / "prompts_de.json"
SR = 16000  # target sample rate

def load_prompts():
    if not PROMPTS_PATH.exists():
        st.error(f"Could not find prompts file at `{PROMPTS_PATH}`. Create it and rerun.")
        st.stop()
    try:
        prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
        if not isinstance(prompts, list) or not all(isinstance(x, str) for x in prompts):
            raise ValueError("prompts_de.json must be a JSON array of strings.")
        return prompts
    except Exception as e:
        st.error(f"Failed to parse `{PROMPTS_PATH}`: {e}")
        st.stop()

def require_ref_audio(idx: int):
    path = REF_DIR / f"{idx:03d}.wav"
    if not path.exists():
        st.error(
            f"Missing reference audio for this prompt: `{path}`.\n\n"
            "Run `python src/tts_gen.py` to generate reference WAVs."
        )
        st.stop()
    return str(path)

def load_audio_from_blob_or_file(audio_blob, file_fallback, sr=SR):
    """
    Returns (y, sr, path_for_transcribe)
    - If mic recording exists, write to a NamedTemporaryFile and return its path for transcribe.
    - Else, load from uploaded file.
    """
    if audio_blob:
        # audio_blob is a BytesIO-like object
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_blob.getvalue())
            tmp_path = tmp.name
        y, _sr = librosa.load(tmp_path, sr=sr, mono=True)
        return y, sr, tmp_path
    elif file_fallback:
        y, _sr = librosa.load(file_fallback, sr=sr, mono=True)
        # For transcribe_with_words we can pass the same file-like object or write temp
        # faster-whisper accepts file paths; create a temp copy to be safe
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y, SR)
            tmp_path = tmp.name

        return y, sr, tmp_path
    else:
        return None, None, None

def render_word_chips(per_word):
    st.write("Per-word feedback (greener = better):")
    spans = []
    from matplotlib.colors import to_rgb  # comes with matplotlib; if not installed, hardcode colors
    def hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0,2,4))

    pr = hex_to_rgb(THEME_PRIMARY)  # target color
    red = (220, 38, 38)            # start color

    for w in per_word:
        s = float(w["score"])  # 0..1
        r = int(red[0] + (pr[0]-red[0]) * s)
        g = int(red[1] + (pr[1]-red[1]) * s)
        b = int(red[2] + (pr[2]-red[2]) * s)
        spans.append(
            f"<span class='chip' style='background-color: rgb({r},{g},{b});'>"
            f"{w['word']} ({int(100*s)}%)</span>"
        )
    st.markdown(" ".join(spans), unsafe_allow_html=True)

# ---------------------------
# UI
# ---------------------------
st.title("Sprich! – German Pronunciation Coach")
st.caption("Record directly in the app or upload a file, then get a 0–100 score with per-word feedback.")

prompts = load_prompts()
if len(prompts) == 0:
    st.warning("Your prompts list is empty. Add some sentences to `data/prompts_de.json`.")
    st.stop()

idx = st.selectbox(
    "Choose a sentence to practice:",
    options=list(range(len(prompts))),
    format_func=lambda i: f"{i:02d} — {prompts[i]}",
    index=0
)
text = prompts[idx]

# Reference audio
ref_path = require_ref_audio(idx)
st.subheader("Reference audio")
st.audio(ref_path, format="audio/wav")

st.divider()

# Mic input + file fallback
st.subheader("Your recording")
audio_blob = st.audio_input("🎙️ Click to record (use your device microphone)")
st.caption("Tip: speak clearly and try to match the reference pacing.")
st.write("Or upload an existing file:")
file_fallback = st.file_uploader("WAV/MP3", type=["wav", "mp3"])

# Score button
if st.button("Score my pronunciation", type="primary"):
    if not (audio_blob or file_fallback):
        st.warning("Please record audio or upload a file first.")
        st.stop()

    # Load reference/user audio
    try:
        y_ref, _ = librosa.load(ref_path, sr=SR, mono=True)
    except Exception as e:
        st.error(f"Failed to load reference audio: {e}")
        st.stop()

    y_usr, _sr, usr_tmp_path = load_audio_from_blob_or_file(audio_blob, file_fallback, sr=SR)
    if y_usr is None:
        st.error("Could not read your audio. Try re-recording or uploading a WAV/MP3.")
        st.stop()

    with st.status("Transcribing & aligning...", expanded=False):
        try:
            ref_words = transcribe_with_words(ref_path, language="de")
            usr_words = transcribe_with_words(usr_tmp_path, language="de")
            user_text = transcribe_text(usr_tmp_path, language="de")
            target_text = text

            score, per_word, meta = score_sentence(
                y_ref, y_usr, SR, ref_words, usr_words, 
                target_text.replace("!", "").replace("?", "").replace(".", "").replace(",", "").split(),
                user_transcript=user_text,
                target_text=target_text
            )
        except Exception as e:
            st.error(f"Whisper transcription failed: {e}")
            st.stop()

    # Results
    st.subheader("Results")
    cols = st.columns(3)
    cols[0].metric("Overall score", f"{score:.0f}/100")
    cols[1].metric("Prosody", f"{meta.get('prosody', 0):.2f}")
    cols[2].metric("Fluency", f"{meta.get('fluency', 0):.2f}")

    render_word_chips(per_word)

    # Coaching tips
    tips_list = tips_for(per_word)
    if tips_list:
        st.subheader("Tips")
        for tip in tips_list:
            st.info(tip)

    # Cleanup temp file created for user audio (if any)
    try:
        if usr_tmp_path and Path(usr_tmp_path).exists():
            os.remove(usr_tmp_path)
    except Exception:
        pass

# Footer
st.divider()
st.caption(
    "MVP uses Whisper word-level timestamps + MFCC/DTW similarity. "
    "For phoneme-level feedback (IPA), add forced alignment later."
)