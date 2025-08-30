# src/tips.py
def tips_for(words):
    out = []
    joined = " ".join(w["word"].lower() for w in words)
    if "ich" in joined:
        out.append("Für /ç/ in 'ich': lächle leicht, Zunge hoch – vermeide [k].")
    if "tag" in joined or "abend" in joined:
        out.append("Endverhärtung: 'Tag' → [taːk], nicht [taːg].")
    if "brötchen" in joined or "müller" in joined:
        out.append("Umlaute: Ü/Ö vorne runden; Zunge bleibt vorn (nicht wie u/o).")
    return out
