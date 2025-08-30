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
from src.align_word import transcribe_with_words
from src.score import score_sentence
from src.tips import tips_for


# ---------------------------
# Page config & helpers
# ---------------------------
st.set_page_config(page_title="Sprich! – German Pronunciation Coach", page_icon="🇩🇪", layout="centered")

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
            librosa.output.write_wav(tmp.name, y, sr)  # deprecated in newer librosa; okay for MVP
            tmp_path = tmp.name
        return y, sr, tmp_path
    else:
        return None, None, None

def render_word_chips(per_word):
    st.write("Per-word feedback (darker = better):")
    # Build a line of colored chips using simple inline HTML
    spans = []
    for w in per_word:
        # w['score'] is 0..1 — map to color (greenish for high, red for low)
        # We'll interpolate between red (low) and green (high) via simple channel math.
        s = float(w["score"])
        r = int(255 * (1 - s))
        g = int(60 + 170 * s)
        b = int(60)
        spans.append(
            f"<span style='background-color: rgb({r},{g},{b}); color: white; "
            f"padding: 4px 10px; margin: 4px; border-radius: 10px; display:inline-block;'>"
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
        except Exception as e:
            st.error(f"Whisper transcription failed: {e}")
            st.stop()

    # Tokenize text (roughly) to pair with word spans
    text_words = (
        text.replace("!", "")
            .replace("?", "")
            .replace(".", "")
            .replace(",", "")
            .split()
    )

    # Score
    with st.status("Scoring pronunciation...", expanded=False):
        try:
            score, per_word, meta = score_sentence(
                y_ref, y_usr, SR, ref_words, usr_words, text_words
            )
        except Exception as e:
            st.error(f"Scoring failed: {e}")
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