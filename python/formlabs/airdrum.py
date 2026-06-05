import asyncio
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from ahrs.filters import Madgwick

from formlabs import CHR, PKT, connect, load_cal, xyz


os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame


SOUNDS_DIR = Path(__file__).parent / ".formlabs" / "sounds"
SOUND_URLS = {
    "left":  "https://tonejs.github.io/audio/drum-samples/CR78/kick.mp3",
    "right": "https://tonejs.github.io/audio/drum-samples/CR78/snare.mp3",
}

SIDE = "right"
BODY_FORWARD = np.array([1.0, 0.0, 0.0])  # board +X = fingertip direction
DRUM_SPLIT_DEG = 25.0                     # each drum sits ±this from neutral
NEUTRAL_DELAY_S = 5.0
SWING_TRIGGER_DPS = 250.0                 # gyro magnitude that counts as a swing start
SWING_REARM_DPS = 60.0                    # must drop below this before next trigger
HIT_DEBOUNCE_S = 0.15


def rotate(q, v):
    w, x, y, z = q
    qv = np.array([x, y, z])
    t = 2 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def yaw_rotate(v, deg):
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


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


async def airdrum():
    paths = ensure_sounds()
    pygame.mixer.init()
    sounds = {name: pygame.mixer.Sound(str(p)) for name, p in paths.items()}

    cal = load_cal(SIDE)
    gbias = xyz(cal["gyroscope"]["bias"])
    abias = xyz(cal["accelerometer"]["bias"])
    moff = xyz(cal["magnetometer"]["offset"])
    mscale = xyz(cal["magnetometer"]["scale"])

    madgwick = Madgwick(gain=0.02)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    last_t = [None]
    start_t = [None]
    drum_dirs = [None]    # filled at neutral capture: {"left": vec, "right": vec}
    last_hit_t = [0.0]
    swing_armed = [True]                # ready to fire on next gyro rising edge

    def on_notify(_, data):
        gx, gy, gz, ax, ay, az, mx, my, mz, *_ = PKT.unpack(data)

        raw_gyr = np.array([gx, gy, gz])
        gyr = (raw_gyr - gbias) * (math.pi / 180.0)
        gyr[2] = -gyr[2]
        acc = np.array([ax, ay, az]) - abias
        mag = (np.array([mx, my, mz]) - moff) * mscale
        mag[1:] = -mag[1:]

        now = time.perf_counter()
        if start_t[0] is None:
            start_t[0] = now
        dt = 0.01 if last_t[0] is None else max(now - last_t[0], 1e-4)
        last_t[0] = now
        madgwick.Dt = dt

        amag = float(np.linalg.norm(acc))
        gmag_dps = math.degrees(float(np.linalg.norm(gyr)))
        clean = 0.90 < amag < 1.10

        if clean and gmag_dps < 2:
            gbias[:] = 0.998 * gbias + 0.002 * raw_gyr

        if drum_dirs[0] is None:
            # Neutral phase: full MARG to lock initial yaw against magnetic north.
            madgwick.gain = 0.05 if clean else 0.0
            q[:] = madgwick.updateMARG(q, gyr=gyr, acc=acc, mag=mag)
            ray = rotate(q, BODY_FORWARD)
            if now - start_t[0] > NEUTRAL_DELAY_S:
                drum_dirs[0] = {
                    "left":  yaw_rotate(ray, -DRUM_SPLIT_DEG),
                    "right": yaw_rotate(ray, +DRUM_SPLIT_DEG),
                }
                print(f"neutral captured — drumming ready (split ±{DRUM_SPLIT_DEG:.0f}°)")
            return

        # Active phase: MARG with low gain only when acc reads clean gravity.
        # Mag at low rate anchors yaw against long-term drift without flipping
        # second-to-second from local-field swings during motion.
        madgwick.gain = 0.02 if clean else 0.0
        q[:] = madgwick.updateMARG(q, gyr=gyr, acc=acc, mag=mag)
        ray = rotate(q, BODY_FORWARD)

        # Trigger on swing onset (rising gyro edge) — fires ~150 ms before peak
        # |acc|, so the sound feels immediate. Use the orientation AT TRIGGER
        # TIME: chop motion hasn't unwound it yet, so it still represents aim.
        if gmag_dps < SWING_REARM_DPS:
            swing_armed[0] = True
        if swing_armed[0] and gmag_dps > SWING_TRIGGER_DPS and now - last_hit_t[0] > HIT_DEBOUNCE_S:
            swing_armed[0] = False
            last_hit_t[0] = now
            which = max(drum_dirs[0], key=lambda n: float(np.dot(ray, drum_dirs[0][n])))
            sounds[which].play()
            print(
                f"HIT  ω={gmag_dps:5.0f}°/s  ray=({ray[0]:+.2f},{ray[1]:+.2f},{ray[2]:+.2f})"
                f"  ->  {which}"
            )

    print(f"connecting to '{SIDE}' stick...")
    print(f"hold hand straight ahead for {int(NEUTRAL_DELAY_S)}s to capture neutral pose")
    async with await connect(SIDE) as ble:
        await ble.start_notify(CHR, on_notify)
        await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(airdrum())
    except KeyboardInterrupt:
        pass
