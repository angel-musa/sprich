# app.py
# Sprich! – German Pronunciation Coach (MVP + classifier + tooltips)

import json, os, tempfile
from pathlib import Path

import numpy as np
import librosa
import streamlit as st
import soundfile as sf
import joblib

from src.score import score_sentence
from src.tips import tips_for
from src.align_word import transcribe_with_words, transcribe_text

# ============ page config must come first ============
st.set_page_config(page_title="Sprich!", page_icon="🇩🇪", layout="centered")

# ---------- theme + CSS ----------
THEME_PRIMARY = st.get_option("theme.primaryColor") or "#58CC02"
THEME_BG      = st.get_option("theme.backgroundColor") or "#FFFFFF"
THEME_BG2     = st.get_option("theme.secondaryBackgroundColor") or "#F2FAEB"
THEME_TEXT    = st.get_option("theme.textColor") or "#1C1E21"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
  background: radial-gradient(1200px 600px at 20% 0%, {THEME_BG2} 0%, {THEME_BG} 40%, {THEME_BG} 100%);
}}
[data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
.block-container {{ max-width: 900px; }}
[data-testid="stMetricValue"] {{ font-weight: 700; }}
.stButton > button {{
  border-radius: 12px;
  padding: 0.6rem 1.1rem;
  font-weight: 600;
}}
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

# ---------- tiny helper for hover info icons ----------
def info_icon(text: str) -> str:
    return f"""<span style="margin-left:6px; cursor: help;" title="{text}">ℹ️</span>"""

# ---------- paths/constants ----------
DATA_DIR = Path("data")
REF_DIR = DATA_DIR / "ref_audio"
PROMPTS_PATH = DATA_DIR / "prompts_de.json"
SR = 16000

# classifier (optional)
FRIC_CLF = None
try:
    FRIC_CLF = joblib.load(DATA_DIR / "fricative_clf.joblib")
except Exception:
    FRIC_CLF = None

# ---------- helpers ----------
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
    if audio_blob:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_blob.getvalue())
            tmp_path = tmp.name
        y, _sr = librosa.load(tmp_path, sr=sr, mono=True)
        return y, sr, tmp_path
    elif file_fallback:
        y, _sr = librosa.load(file_fallback, sr=sr, mono=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y, sr)
            tmp_path = tmp.name
        return y, sr, tmp_path
    else:
        return None, None, None

def render_word_chips(per_word):
    st.write("Per-word feedback (**greener = better, redder = worse**):")
    spans = []
    def hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0,2,4))
    pr = hex_to_rgb(THEME_PRIMARY)
    red = (220, 38, 38)
    for w in per_word:
        s = float(w["score"])
        r = int(red[0] + (pr[0]-red[0]) * s)
        g = int(red[1] + (pr[1]-red[1]) * s)
        b = int(red[2] + (pr[2]-red[2]) * s)
        spans.append(
            f"<span class='chip' style='background-color: rgb({r},{g},{b});'>"
            f"{w['word']} ({int(100*s)}%)</span>"
        )
    st.markdown(" ".join(spans), unsafe_allow_html=True)

def _delta_safe(M):
    T = M.shape[1]
    if T < 3:
        return np.zeros_like(M)
    w = min(9, T-1 if (T-1) % 2 == 1 else T-2)
    if w < 3:
        return np.zeros_like(M)
    return librosa.feature.delta(M, width=w)

def _clf_features(y, sr=22050):
    if sr != 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050); sr = 22050
    min_len = int(0.3 * sr)
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))
    y, _ = librosa.effects.trim(y, top_db=25)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, n_fft=1024, hop_length=256)
    L = librosa.power_to_db(S + 1e-6)
    D1 = _delta_safe(L)
    D2 = _delta_safe(L)
    def tstats(A): return np.hstack([A.mean(axis=1), A.std(axis=1)])
    feat = np.hstack([tstats(L), tstats(D1), tstats(D2)]).astype(np.float32)
    return feat.reshape(1, -1)


def classify_fricative(y, sr):
    if FRIC_CLF is None:
        return None, None
    X = _clf_features(y, sr)
    model = FRIC_CLF["model"]
    labels = FRIC_CLF["classes"]
    pred_idx = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    probs_dict = {lbl: float(p) for lbl, p in zip(labels, probs)}
    return labels[pred_idx], probs_dict

# ---------- UI ----------
st.title("Sprich! – German Pronunciation Coach")
st.caption("Record or upload your attempt; get a 0–100 score with word feedback. Includes an ML phoneme detector for ich-/ach-Laut.")

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

# reference
ref_path = require_ref_audio(idx)
st.subheader("Reference audio")
st.audio(ref_path, format="audio/wav")

st.divider()

# mic + upload
st.subheader("Your recording")
audio_blob = st.audio_input("🎙️ Click to record (use your device microphone)")
st.caption("Tip: speak clearly and try to match the reference pacing.")
st.write("Or upload an existing file:")
file_fallback = st.file_uploader("WAV/MP3", type=["wav", "mp3"])

# action
if st.button("Score my pronunciation", type="primary"):
    if not (audio_blob or file_fallback):
        st.warning("Please record audio or upload a file first.")
        st.stop()

    try:
        y_ref, _ = librosa.load(ref_path, sr=SR, mono=True)
    except Exception as e:
        st.error(f"Failed to load reference audio: {e}")
        st.stop()

    y_usr, _sr, usr_tmp_path = load_audio_from_blob_or_file(audio_blob, file_fallback, sr=SR)
    if y_usr is None:
        st.error("Could not read your audio. Try re-recording or uploading a WAV/MP3.")
        st.stop()

    # transcribe + score
    with st.status("Transcribing & scoring...", expanded=False):
        try:
            ref_words = transcribe_with_words(ref_path, language="de")
            usr_words = transcribe_with_words(usr_tmp_path, language="de")
            user_text = transcribe_text(usr_tmp_path, language="de")
            target_text = text
            text_words = (
                target_text.replace("!", "").replace("?", "").replace(".", "").replace(",", "").split()
            )
            score, per_word, meta = score_sentence(
                y_ref, y_usr, SR, ref_words, usr_words,
                text_words,
                user_transcript=user_text,
                target_text=target_text
            )
        except Exception as e:
            st.error(f"Transcription/scoring failed: {e}")
            st.stop()

    # ---------- RESULTS ----------
    st.markdown("### Results")  # headings via markdown, not subheader

    c1, c2, c3 = st.columns(3)

    # Overall
    c1.markdown("**Overall score** " + info_icon(
        "0–100 composite = 70% pronunciation similarity (DTW on log-mel), 20% prosody (pitch contour), 10% fluency (pauses/energy), scaled by text accuracy."
    ), unsafe_allow_html=True)
    c1.metric(label="", value=f"{score:.0f}/100")  # empty label

    # Prosody
    c2.markdown("**Prosody** " + info_icon(
        "How closely your pitch contour matches the reference (intonation)."
    ), unsafe_allow_html=True)
    c2.metric(label="", value=f"{meta.get('prosody', 0):.2f}")

    # Fluency
    c3.markdown("**Fluency** " + info_icon(
        "Signal-based fluency: fewer/shorter pauses and steady energy = higher score."
    ), unsafe_allow_html=True)
    c3.metric(label="", value=f"{meta.get('fluency', 0):.2f}")


    render_word_chips(per_word)

    # ---------- TIPS ----------
    tips_list = tips_for(per_word)
    if tips_list:
        st.markdown("### Tips " + info_icon("Targeted coaching suggestions based on low-scoring words/phonemes."), unsafe_allow_html=True)
        for tip in tips_list:
            st.info(tip)

    # ---------- CLASSIFIER ----------
    st.divider()
    st.markdown("### Phoneme classifier (ML) " + info_icon(
    "Tiny supervised model trained on German TTS + samples. It detects /ç/ (ich-Laut), /x/ (ach-Laut), or /k/. "
    "Phonemes are the smallest units of sound that distinguish words."
    ), unsafe_allow_html=True)

    if FRIC_CLF is None:
        st.caption("Classifier model not found. Train it with `python src/train_fricative_classifier.py` (saves to `data/fricative_clf.joblib`).")
    else:
        label, probs = classify_fricative(y_usr, SR)
        if label is None:
            st.caption("Could not classify (no audio).")
        else:
            st.metric("Predicted phoneme", label)
            cols = st.columns(len(probs))
            for (lbl, p), c in zip(probs.items(), cols):
                with c:
                    st.write(lbl)
                    st.progress(min(max(p, 0.0), 1.0), text=f"{p:.2f}")

            low_text = text.lower()
            if any(w in low_text for w in ["ich","nicht","milch","licht","pech","becher","mädchen","bisschen","china"]):
                st.caption("This prompt likely contains **/ç/** (ich-Laut).")
            if any(w in low_text for w in ["ach","bach","dach","sache","rauch","buch","kuchen","fuchs","auch"]):
                st.caption("This prompt likely contains **/x/** (ach-Laut).")

    # cleanup
    try:
        if usr_tmp_path and Path(usr_tmp_path).exists():
            os.remove(usr_tmp_path)
    except Exception:
        pass

# footer
st.divider()
st.caption(
    "Scoring: log-mel DTW + duration + prosody + improved fluency; WER-based scaling. "
    "Classifier: logistic regression on log-mel+deltas for /ç/ vs /x/ vs /k/."
)
