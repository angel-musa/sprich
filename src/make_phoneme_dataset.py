# src/make_phoneme_dataset.py
# Generate a tiny dataset for /ç/ (ich-Laut), /x/ (ach-Laut), and /k/ controls.
# Uses Coqui TTS "Thorsten VITS" (stable) and cleans each clip to ~1.0–1.2s.

from pathlib import Path
import json
import numpy as np
import soundfile as sf
import librosa
import random

# ----------------------------
# Config (tweak as you like)
# ----------------------------
OUT_DIR = Path("data/phoneme_ds")
SAMPLE_RATE = 22050
MAX_SEC = 1.2       # hard cap per clip (seconds)
MIN_SEC = 0.6       # ensure not too short (pad if needed)
TOP_DB = 25         # trimming threshold
VARIANTS_PER_WORD = 2   # 1 = single file per word; >1 adds speed/pitch/noise
LENGTH_SCALES = [0.9, 1.0]          # <1 faster, >1 slower (VITS)
PITCH_STEPS = [0, +1]               # semitones; keep small to avoid artifacts
NOISE_SNR_DB = [None, 30]           # None means no noise; else SNR in dB

# Words where 'ch' -> /ç/ (after front vowels / consonants)
ICH_WORDS = [
    "ich", "nicht", "Milch", "Licht", "Fichte",
    "Pech", "Becher", "Mädchen", "bisschen", "China"
]
# Words where 'ch' -> /x/ (after back vowels a, o, u, au)
ACH_WORDS = [
    "ach", "Bach", "Dach", "Sache", "Rauch",
    "Buch", "Kuchen", "Fuchs", "auch"
]
# /k/ control words (no overlap with above)
K_WORDS = [
    "Ike", "Nika", "Baka", "Baku", "Koma",
    "Akku", "Backe", "Balkon", "Mücke"
]

# ----------------------------
# TTS & audio helpers
# ----------------------------
def get_tts():
    # Thorsten VITS is non-autoregressive and avoids Tacotron “looping”
    from TTS.api import TTS
    return TTS("tts_models/de/thorsten/vits")

def synth(tts, text: str, length_scale: float = 0.9):
    # VITS supports length_scale to control speaking rate
    wav = np.asarray(tts.tts(text=text, length_scale=length_scale), dtype=np.float32)
    return wav, SAMPLE_RATE

def normalize_and_clip(y: np.ndarray, sr: int) -> np.ndarray:
    # 1) Trim leading/trailing silence
    y_trim, _ = librosa.effects.trim(y, top_db=TOP_DB)
    # 2) Remove long internal silences
    intervals = librosa.effects.split(y_trim, top_db=TOP_DB)
    if len(intervals) > 0:
        parts = [y_trim[s:e] for s, e in intervals]
        y_v = np.concatenate(parts, axis=0)
    else:
        y_v = y_trim
    # 3) Cap duration
    max_len = int(MAX_SEC * sr)
    if len(y_v) > max_len:
        y_v = y_v[:max_len]
    # 4) Ensure minimum duration
    min_len = int(MIN_SEC * sr)
    if len(y_v) < min_len:
        y_v = np.pad(y_v, (0, max(0, min_len - len(y_v))))
    # 5) Normalize peak
    peak = float(np.max(np.abs(y_v)) + 1e-6)
    y_v = 0.97 * y_v / peak
    return y_v

def add_noise(y: np.ndarray, snr_db: float) -> np.ndarray:
    if snr_db is None:
        return y
    rms = np.sqrt(np.mean(y**2)) + 1e-9
    n = np.random.randn(len(y)).astype(np.float32)
    n = n / (np.sqrt(np.mean(n**2)) + 1e-9)
    noise_gain = rms / (10 ** (snr_db / 20.0))
    return y + noise_gain * n

def variant_params(v_idx: int):
    # pick params cyclically so VARIANTS_PER_WORD can be any small int
    ls = LENGTH_SCALES[v_idx % len(LENGTH_SCALES)]
    ps = PITCH_STEPS[v_idx % len(PITCH_STEPS)]
    snr = NOISE_SNR_DB[v_idx % len(NOISE_SNR_DB)]
    return ls, ps, snr

def apply_variant(y: np.ndarray, sr: int, length_scale: float, pitch_steps: int, snr_db):
    # length_scale handled at TTS synth time for better quality;
    # here we only apply pitch and noise if requested.
    if pitch_steps != 0:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_steps)
    y = add_noise(y, snr_db)
    return y

# ----------------------------
# Main
# ----------------------------
def generate_class(tts, label: str, words: list[str], out_dir: Path):
    out = out_dir / label
    out.mkdir(parents=True, exist_ok=True)
    uid = 0
    for w in words:
        # base render for each length_scale we plan to use (keeps quality high)
        for v in range(VARIANTS_PER_WORD):
            ls, ps, snr = variant_params(v)
            y, sr = synth(tts, w, length_scale=ls)
            y = apply_variant(y, sr, length_scale=ls, pitch_steps=ps, snr_db=snr)
            y = normalize_and_clip(y, sr)
            sf.write(out / f"{uid:04d}_{w}_v{v}.wav", y, sr)
            uid += 1

def main():
    random.seed(42)
    tts = get_tts()
    if OUT_DIR.exists():
        print(f"[info] Writing into: {OUT_DIR.resolve()}")
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_class(tts, "ich", ICH_WORDS, OUT_DIR)
    generate_class(tts, "ach", ACH_WORDS, OUT_DIR)
    generate_class(tts, "k",   K_WORDS,   OUT_DIR)

    (OUT_DIR / "meta.json").write_text(
        json.dumps({"classes": ["ich", "ach", "k"]}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print("[done] Dataset ready.")

if __name__ == "__main__":
    main()
