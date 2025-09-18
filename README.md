# 🇩🇪 Sprich! – German Pronunciation Coach

Sprich! (German for *Speak!*) is an interactive web app that helps learners **practice German pronunciation**.  
It combines **speech recognition**, **acoustic similarity scoring**, and a custom **phoneme classifier** for the tricky *ich-Laut* (/ç/) vs *ach-Laut* (/x/).

Built with **Python + Streamlit + ML**.
Check it out here: https://sprich.streamlit.app/

---

## ✨ Features
- 🎙️ **Record in browser** or upload audio (WAV/MP3).  
- 🔊 **Reference audio** generated from high-quality German TTS.  
- 📝 **ASR transcription & alignment** (Whisper) to align learner vs reference word-by-word.  
- 📊 **Scoring system**
  - **Overall score (0–100)**: weighted blend of pronunciation similarity, prosody, and fluency.  
  - **Prosody**: how well the pitch contour matches the reference.  
  - **Fluency**: fewer pauses and smoother energy = higher score.  
  - **WER penalty**: penalizes large transcription errors.  
- 🌈 **Per-word feedback chips** (redder = weaker, greener = stronger).  
- 💡 **Tips engine**: surface targeted articulation advice (e.g. *avoid [k] for /ç/*).  
- 🧠 **Phoneme classifier (ML)**: logistic regression trained on synthetic + sample data to detect `/ç/`, `/x/`, or `/k/`.  
- 🎨 **Custom theme** with gradient background and tooltip info icons.

---

## 🏗️ Architecture
- **Frontend/UI**: [Streamlit](https://streamlit.io/)  
- **Speech alignment**: [faster-whisper](https://github.com/guillaumekln/faster-whisper) for ASR & timestamps  
- **Audio features**: [librosa](https://librosa.org/)  
- **Similarity**: log-mel + dynamic time warping (DTW)  
- **Tips**: rule-based heuristics from per-word scores  
- **Classifier**: scikit-learn logistic regression on log-mel + delta features  

---

## 🚀 Quickstart

### 1. Clone repo
```bash
git clone https://github.com/YOURNAME/sprich.git
cd sprich
