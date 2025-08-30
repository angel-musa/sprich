# src/score.py
import numpy as np, librosa
import sys

def _mfcc(y, sr): 
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

def _dtw_sim(mfcc_ref, mfcc_usr):
    D, wp = librosa.sequence.dtw(mfcc_ref, mfcc_usr, metric="cosine")
    dist = D[-1, -1] / len(wp)
    # map distance → [0..1] (tune tau)
    tau = 3.0
    sim = 1.0 / (1.0 + dist / tau)
    return sim

def extract_clip(y, sr, t0, t1):
    i0, i1 = int(t0*sr), int(t1*sr)
    i0 = max(i0,0); i1 = min(i1, len(y))
    return y[i0:i1]

def pitch_slope(y, sr):
    f0, _, _ = librosa.pyin(y, fmin=75, fmax=350, sr=sr)
    f0 = f0[~np.isnan(f0)]
    if len(f0) < 5: return 0.0
    return float(np.polyfit(np.arange(len(f0)), f0, 1)[0])

def score_sentence(y_ref, y_usr, sr, ref_words, usr_words, text_words):
    # naive word alignment by text order length; robust enough for MVP when users read the prompt
    n = min(len(ref_words), len(usr_words), len(text_words))
    per_word = []
    seg_scores = []
    for i in range(n):
        rw, uw, token = ref_words[i], usr_words[i], text_words[i]
        y_r = extract_clip(y_ref, sr, rw["start"], rw["end"])
        y_u = extract_clip(y_usr, sr, uw["start"], uw["end"])
        if len(y_r) < sr*0.05 or len(y_u) < sr*0.05:
            s = 0.0
        else:
            mr, mu = _mfcc(y_r, sr), _mfcc(y_u, sr)
            s = _dtw_sim(mr, mu)
        # duration similarity
        dur_ref, dur_usr = rw["end"]-rw["start"], uw["end"]-uw["start"]
        dur_sim = 1.0 - min(1.0, abs(dur_ref - dur_usr) / max(0.2, dur_ref))
        word_score = 0.8*s + 0.2*dur_sim
        per_word.append({"word": token, "score": float(word_score)})
        seg_scores.append(word_score)

    # prosody/fluency
    ps_ref = pitch_slope(y_ref, sr); ps_usr = pitch_slope(y_usr, sr)
    prosody = 1.0 - min(1.0, abs(ps_ref - ps_usr)/50.0)

    # pauses: penalize long silences in user audio (very rough)
    rms = librosa.feature.rms(y=y_usr).mean()
    fluency = 1.0 if rms>0 else 0.7

    overall = 100.0 * (0.7*np.mean(seg_scores) + 0.2*prosody + 0.1*fluency)
    return float(np.clip(overall, 0, 100)), per_word, {"prosody": float(prosody), "fluency": float(fluency)}
