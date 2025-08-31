from pathlib import Path
import json, numpy as np, librosa, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

SR = 22050

def features(path: Path) -> np.ndarray:
    y, sr = librosa.load(path, sr=SR, mono=True)
    # trim silence
    y, _ = librosa.effects.trim(y, top_db=25)
    if len(y) < 0.3*sr:  # pad very short clips
        y = np.pad(y, (0, int(0.3*sr)-len(y)))
    # log-mel + deltas
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, n_fft=1024, hop_length=256)
    L = librosa.power_to_db(S + 1e-6)
    D1 = librosa.feature.delta(L)
    D2 = librosa.feature.delta(L, order=2)
    # summarise by mean + std
    def tstats(A): return np.hstack([A.mean(axis=1), A.std(axis=1)])
    feat = np.hstack([tstats(L), tstats(D1), tstats(D2)])
    return feat.astype(np.float32)

def load_dataset(root=Path("data/phoneme_ds")):
    meta = json.loads((root/"meta.json").read_text(encoding="utf-8"))
    classes = meta["classes"]
    X, y = [], []
    for label in classes:
        for wav in (root/label).glob("*.wav"):
            X.append(features(wav))
            y.append(classes.index(label))
    return np.vstack(X), np.array(y), classes

def main():
    X, y, classes = load_dataset()
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=300, multi_class="multinomial"))
    ])
    # quick cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(f"5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    # fit final
    clf.fit(X, y)
    joblib.dump({"model": clf, "classes": classes}, "data/fricative_clf.joblib")
    print("[done] Saved model to data/fricative_clf.joblib")

if __name__ == "__main__":
    main()
