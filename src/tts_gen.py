# src/tts_gen.py
from pathlib import Path
import json, soundfile as sf
import numpy as np
from TTS.api import TTS  # heavy import after env ok

def main():
    tts = TTS("tts_models/de/thorsten/tacotron2-DDC")  # small, decent
    prompts = json.load(open("data/prompts_de.json","r",encoding="utf-8"))
    out = Path("data/ref_audio"); out.mkdir(parents=True, exist_ok=True)
    for i, txt in enumerate(prompts):
        wav = tts.tts(text=txt)  # returns np.float32 22kHz-ish
        sf.write(out/f"{i:03d}.wav", np.array(wav), 22050)

if __name__ == "__main__":
    main()
