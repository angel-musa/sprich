# src/score.py
import numpy as np, librosa
from jiwer import wer

def _delta_safe(M):
    """Delta with adaptive odd width strictly < T; zeros if too short."""
    T = M.shape[1]
    if T < 3:
        return np.zeros_like(M)
    # pick the largest odd width strictly less than T, capped at 9
    w = min(9, T-1 if (T-1) % 2 == 1 else T-2)
    if w < 3:
        return np.zeros_like(M)
    return librosa.feature.delta(M, width=w)

def _logmel(y, sr):
    # ensure at least ~0.2s so we get a few frames
    min_len = int(0.2 * sr)
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, n_fft=1024, hop_length=256)
    L = librosa.power_to_db(S + 1e-10)
    d1 = _delta_safe(L)
    d2 = _delta_safe(L)
    return np.vstack([L, d1, d2])

def extract_clip(y, sr, t0, t1):
    i0, i1 = int(max(0.0, t0)*sr), int(max(0.0, t1)*sr)
    i0 = max(i0, 0); i1 = min(i1, len(y))
    if i1 <= i0:
        return np.zeros(int(0.05*sr), dtype=float)
    return y[i0:i1]

def _voicing_fraction(y, sr):
    f0, _, _ = librosa.pyin(y, fmin=75, fmax=350, sr=sr)
    if f0 is None: return 0.0
    voiced = np.isfinite(f0)
    return float(voiced.mean()) if voiced.size else 0.0

def pitch_slope(y, sr):
    f0, _, _ = librosa.pyin(y, fmin=75, fmax=350, sr=sr)
    if f0 is None: return 0.0
    f0 = f0[~np.isnan(f0)]
    if len(f0) < 5: return 0.0
    return float(np.polyfit(np.arange(len(f0)), f0, 1)[0])

def _dtw_sim(F_ref, F_usr):
    D, wp = librosa.sequence.dtw(F_ref, F_usr, metric="cosine")
    if len(wp) == 0: return 0.0
    dist = float(D[-1, -1] / len(wp))
    k = 1.2
    return float(np.clip(np.exp(-dist / k), 0.0, 1.0))

def _utterance_sim(y_ref, y_usr, sr):
    return _dtw_sim(_logmel(y_ref, sr), _logmel(y_usr, sr))

def _duration_sim(d_ref, d_usr):
    if d_ref <= 0.01: return 0.0
    rel = abs(d_ref - d_usr) / d_ref
    return float(np.clip(1.0 - (rel / 0.3), 0.0, 1.0))

def fluency_score(y, sr):
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    thr = max(0.02, float(np.median(rms) * 0.3))
    voiced_frac = float((rms > thr).mean())
    y_trim, _ = librosa.effects.trim(y, top_db=30)
    trim_ratio = float(len(y_trim) / max(1, len(y)))
    return float(np.clip(0.5*voiced_frac + 0.5*trim_ratio, 0.0, 1.0))

def score_sentence(y_ref, y_usr, sr, ref_words, usr_words, text_words,
                   user_transcript=None, target_text=None):
    n = min(len(ref_words), len(usr_words), len(text_words))
    per_word, seg_scores = [], []

    if n > 0:
        for i in range(n):
            rw, uw, token = ref_words[i], usr_words[i], text_words[i]
            y_r = extract_clip(y_ref, sr, rw["start"], rw["end"])
            y_u = extract_clip(y_usr, sr, uw["start"], uw["end"])

            if len(y_r) < sr*0.05 or len(y_u) < sr*0.05:
                s = 0.0
            else:
                s = _dtw_sim(_logmel(y_r, sr), _logmel(y_u, sr))

            dur_sim = _duration_sim(rw["end"]-rw["start"], uw["end"]-uw["start"])
            word_score = float(np.clip(0.7*s + 0.3*dur_sim, 0.0, 1.0))
            per_word.append({"word": token, "score": word_score})
            seg_scores.append(word_score)

    if len(seg_scores) == 0:
        s_all = _utterance_sim(y_ref, y_usr, sr)
        per_word = [{"word": (w if text_words else "Gesamte Äußerung"), "score": s_all}
                    for w in (text_words or ["Gesamte Äußerung"])]
        seg_mean = s_all
    else:
        seg_mean = float(np.mean(seg_scores))

    ps_ref = pitch_slope(y_ref, sr); ps_usr = pitch_slope(y_usr, sr)
    voicing = _voicing_fraction(y_usr, sr)
    prosody_raw = 1.0 - min(1.0, abs(ps_ref - ps_usr)/50.0)
    prosody = float(prosody_raw * np.clip((voicing - 0.2)/0.6, 0.0, 1.0))

    fluency = fluency_score(y_usr, sr)

    text_scale = 1.0
    if user_transcript and target_text:
        hyp = user_transcript.strip().lower()
        ref = target_text.strip().lower()
        w = float(min(1.0, wer(ref, hyp)))
        text_scale = float(1.0 - 0.4*w)

    base = (0.72*seg_mean + 0.18*prosody + 0.10*fluency)
    overall = float(np.clip(100.0 * base * text_scale, 0.0, 100.0))

    return overall, per_word, {
        "prosody": float(prosody),
        "fluency": float(fluency),
        "voicing": float(voicing),
        "text_scale": float(text_scale),
    }
