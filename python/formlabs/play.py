import asyncio
import os
import time
import urllib.request
from collections import deque
from pathlib import Path

import joblib
import numpy as np

from formlabs import CHR, PKT, connect

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame


MODEL_PATH = Path(__file__).parent / ".formlabs" / "model.pkl"
SOUNDS_DIR = Path(__file__).parent / ".formlabs" / "sounds"
SOUND_URLS = {
    "left":  "https://tonejs.github.io/audio/drum-samples/CR78/kick.mp3",
    "right": "https://tonejs.github.io/audio/drum-samples/CR78/snare.mp3",
}

SIDE = "right"
PROB_THRESHOLD = 0.7
HIT_DEBOUNCE_S = 0.15


def ensure_sounds():
    SOUNDS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in SOUND_URLS.items():
        path = SOUNDS_DIR / f"{name}.mp3"
        if not path.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, path)
        paths[name] = path
    return paths


async def play():
    if not MODEL_PATH.exists():
        raise SystemExit(f"{MODEL_PATH} not found — run train.py first")

    sound_paths = ensure_sounds()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    sounds = {name: pygame.mixer.Sound(str(p)) for name, p in sound_paths.items()}

    bundle = joblib.load(MODEL_PATH)
    model, scaler = bundle["model"], bundle["scaler"]
    WINDOW = bundle["window"]
    CLASSES = bundle["classes"]

    buf = deque(maxlen=WINDOW)
    last_hit_t = [0.0]

    def on_notify(_, data):
        buf.append(list(PKT.unpack(data)[:9]))
        if len(buf) < WINDOW:
            return

        x = scaler.transform(np.asarray(buf, dtype=np.float32).reshape(1, -1))
        probs = model.predict_proba(x)[0]

        now = time.perf_counter()
        cls = int(np.argmax(probs))
        p = float(probs[cls])
        if cls != 0 and p > PROB_THRESHOLD and now - last_hit_t[0] > HIT_DEBOUNCE_S:
            last_hit_t[0] = now
            name = CLASSES[cls]
            sounds[name].play()
            print(f"HIT  {name:>5}  p={p:.2f}")

    async with await connect(SIDE) as ble:
        await ble.start_notify(CHR, on_notify)
        print(f"playing — Ctrl-C to stop  (threshold p>{PROB_THRESHOLD})")
        await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(play())
    except KeyboardInterrupt:
        pass
