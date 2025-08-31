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
    seg_list = list(segments)
    for seg in seg_list:
        if seg.words:
            for w in seg.words:
                if w.start is not None and w.end is not None:
                    words.append({"text": (w.word or "").strip(), "start": float(w.start), "end": float(w.end)})

    if not words and seg_list:
        start = min(s.start for s in seg_list if s.start is not None)
        end   = max(s.end   for s in seg_list if s.end   is not None)
        if start is not None and end is not None and end > start:
            words = [{"text": "UTTERANCE", "start": float(start), "end": float(end)}]
    return words

def transcribe_text(path, language="de"):
    """Return a plain transcript string (no timestamps) for WER penalty."""
    model = load_model()
    segments, info = model.transcribe(path, language=language, vad_filter=True)
    segs = list(segments)
    return " ".join(s.text.strip() for s in segs if s.text)
