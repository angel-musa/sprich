# src/align_word.py
from faster_whisper import WhisperModel

_model = None
def load_model(size="small"):
    global _model
    if _model is None:
        _model = WhisperModel(size, device="cpu", compute_type="int8")
    return _model

def transcribe_with_words(path, language="de"):
    model = load_model()
    segments, info = model.transcribe(path, language=language, word_timestamps=True, vad_filter=True)
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({"text": w.word, "start": float(w.start), "end": float(w.end)})
    return words
