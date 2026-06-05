import asyncio
import json
import math
import mido
import struct
import sys
import time
from pathlib import Path

import numpy as np
from ahrs.filters import Madgwick
from bleak import BleakClient, BleakScanner
from pythonosc import udp_client

# ==========================================
# =====           CONFIG               =====
# ==========================================
GLOVE_SIDE = "right"  # "left" or "right"
GLOVE_SVC  = "fa9b1d2c-3e4f-4a5b-9c6d-7e8f9a0b1c2d"
GLOVE_CHR  = "fa9b1d2c-3e4f-4a5b-9c6d-7e8f9a0b1c2e"
GLOVE_PKT  = struct.Struct("<9f4b")
CAL_PATH   = Path(__file__).parent.parent / "python" / "formlabs" / ".formlabs" / "calibration.json"

osc_client    = udp_client.SimpleUDPClient("127.0.0.1", 8000)
IAC_PORT_NAME = 'IAC illeszt≈ëprogram 1. busz'
WARMUP_PKTS   = 50
NUM_TRACKS    = 8

ACTIVE_SCALE_MODE = "aeolian"
SCALE_LIBRARY = {
    "aeolian": [0, 3, 7, 10],
    "ionian":  [0, 4, 7, 11],
    "dorian":  [0, 3, 7, 9],
}
ROOT_NOTE          = 60
intervals          = SCALE_LIBRARY.get(ACTIVE_SCALE_MODE, SCALE_LIBRARY["aeolian"])
active_note_states = [False, False, False, False]

# Tempo (pitch axis on synth): 1 BPM / 5 deg  →  ±9 BPM over the 90° window
TEMPO_BASE = 120.0
TEMPO_HALF = 9.0

# EMA alpha for choir smoothing (lower = smoother, prevents jumps)
CHOIR_ALPHA = 0.04


# ==========================================
# =====        SLIDING WINDOW          =====
# ==========================================

class SlidingWindow:
    """90° window that slides when the angle hits a boundary."""
    def __init__(self, center=0.0, half=45.0):
        self.center = center
        self.half   = half

    def update(self, angle):
        if angle > self.center + self.half:
            self.center = angle - self.half
        elif angle < self.center - self.half:
            self.center = angle + self.half
        return max(0.0, min(1.0, (angle - self.center + self.half) / (2 * self.half)))


# ==========================================
# =====       PRESET DEFINITIONS       =====
# ==========================================
# n/m → fx n, param m on the preset's track.
# Stereo encoder (spatial audio) is always FX 4 on each track.
# scale: output range compressed around 0.5 (1.0 = full 0-1, 0.3 = 0.35-0.65)

def _osc(track, fx, param):
    return f"/track/{track}/fx/{fx}/fxparam/{param}/value"

def _p(track, fx, param, scale=1.0):
    return {"addr": _osc(track, fx, param), "scale": scale}

PRESETS = {
    1: {
        "name":  "Drum",
        "track": 1,
        "roll":  _p(1, 1, 598),         # cutoff
        "pitch": _p(1, 1, 599),         # reso
        "yaw":   _p(1, 6, 8),           # valhalla density
    },
    2: {
        "name":        "Polymax",
        "track":       2,
        "roll":        _p(2, 1, 5),     # cutoff  ← confirm param number
        "yaw":         _p(2, 1, 6),     # reso    ← confirm param number
        "pitch_tempo": True,             # pitch → global REAPER tempo (±9 BPM)
    },
    3: {
        "name":   "Choir",
        "track":  3,
        "smooth": True,                  # EMA smoothing to prevent jumps
        "yaw":    _p(3, 1, 15),         # syllable x
        "pitch":  _p(3, 1, 16),         # syllable y
        "roll":   _p(3, 7, 4, 0.3),    # astro mix (tamed to 30% range)
    },
}


# ==========================================
# =====    CALIBRATION / ORIENTATION   =====
# ==========================================

def _xyz(d):
    return np.array([d["x"], d["y"], d["z"]], dtype=float)


def load_cal():
    cal = json.loads(CAL_PATH.read_text())
    return cal[GLOVE_SIDE]


def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def q_inv(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_to_euler(q):
    w, x, y, z = q
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp  = 2*(w*y - z*x)
    pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def send_spatial(track, q_rel):
    w, x, y, z = q_rel
    osc_client.send_message(_osc(track, 4, 3), float((w + 1) / 2))
    osc_client.send_message(_osc(track, 4, 4), float((x + 1) / 2))
    osc_client.send_message(_osc(track, 4, 5), float((y + 1) / 2))
    osc_client.send_message(_osc(track, 4, 6), float((-z + 1) / 2))  # invert left-right


# ==========================================
# =====             MAIN               =====
# ==========================================

async def main(active):
    global active_note_states

    cal    = load_cal()
    gbias  = _xyz(cal["gyroscope"]["bias"])
    abias  = _xyz(cal["accelerometer"]["bias"])
    moff   = _xyz(cal["magnetometer"]["offset"])
    mscale = _xyz(cal["magnetometer"]["scale"])

    madgwick  = Madgwick(gain=0.02)
    q         = np.array([1.0, 0.0, 0.0, 0.0])
    last_t    = [None]
    warmup    = [0]
    q_ref     = [None]

    win_roll  = SlidingWindow()  # centered at 0°
    win_pitch = SlidingWindow()
    win_yaw   = SlidingWindow()

    smooth    = {}  # addr → EMA value, lazily initialized on first send

    def arm_tracks():
        active_tracks = {PRESETS[p]["track"] for p in active}
        for n in range(1, NUM_TRACKS + 1):
            osc_client.send_message(f"/track/{n}/recarm", 1 if n in active_tracks else 0)

    def send_param(m, val, use_smooth):
        """Send a single FX param. m = {"addr": ..., "scale": ...}"""
        scaled = max(0.0, min(1.0, 0.5 + (val - 0.5) * m["scale"]))
        addr = m["addr"]
        if use_smooth:
            if addr not in smooth:
                smooth[addr] = scaled  # initialize at actual value — no jump
            smooth[addr] = CHOIR_ALPHA * scaled + (1 - CHOIR_ALPHA) * smooth[addr]
            osc_client.send_message(addr, float(smooth[addr]))
        else:
            osc_client.send_message(addr, float(scaled))

    print(f"Scanning for '{GLOVE_SIDE}' glove...", flush=True)
    device = await BleakScanner.find_device_by_filter(
        lambda _, advert: (
            GLOVE_SVC in (u.lower() for u in advert.service_uuids)
            and (advert.local_name or "").lower() == GLOVE_SIDE
        ),
        timeout=20.0,
    )
    if device is None:
        print(f"FATAL: Could not find '{GLOVE_SIDE}' glove.")
        sys.exit(1)

    try:
        midi_out = mido.open_output(IAC_PORT_NAME)
        print(f"SUCCESS: Bound to MIDI -> {IAC_PORT_NAME}")
    except OSError:
        print(f"FATAL: Could not find MIDI port '{IAC_PORT_NAME}'.")
        sys.exit(1)

    def on_notify(_, data):
        global active_note_states
        try:
            gx, gy, gz, ax, ay, az, mx, my, mz, b0, b1, b2, b3 = GLOVE_PKT.unpack(data)
            buttons = (b0, b1, b2, b3)

            raw_gyr = np.array([gx, gy, gz])
            gyr = (raw_gyr - gbias) * (math.pi / 180.0)
            gyr[2] = -gyr[2]
            acc = np.array([ax, ay, az]) - abias
            mag = (np.array([mx, my, mz]) - moff) * mscale
            mag[1:] = -mag[1:]

            now = time.perf_counter()
            dt  = 0.01 if last_t[0] is None else max(now - last_t[0], 1e-4)
            last_t[0] = now
            madgwick.Dt = dt

            if 0.95 < np.linalg.norm(acc) < 1.05 and np.linalg.norm(gyr) < math.radians(2):
                gbias[:] = 0.998 * gbias + 0.002 * raw_gyr

            q[:] = madgwick.updateMARG(q, gyr=gyr, acc=acc, mag=mag)

            warmup[0] += 1
            if warmup[0] < WARMUP_PKTS:
                return
            if q_ref[0] is None:
                q_ref[0] = q.copy()
                arm_tracks()
                print("\nForward reference locked.\n", flush=True)
                return

            q_rel = q_mul(q_inv(q_ref[0]), q)
            roll, pitch, yaw = quat_to_euler(q_rel)  # relative to reference — windows start at 0

            v_roll  = win_roll.update(roll)
            v_pitch = win_pitch.update(pitch)
            v_yaw   = win_yaw.update(yaw)

            for pid in active:
                p          = PRESETS[pid]
                use_smooth = p.get("smooth", False)

                if "roll"  in p: send_param(p["roll"],  v_roll,  use_smooth)
                if "pitch" in p: send_param(p["pitch"], v_pitch, use_smooth)
                if "yaw"   in p: send_param(p["yaw"],   v_yaw,   use_smooth)

                if p.get("pitch_tempo"):
                    bpm = TEMPO_BASE + (v_pitch - 0.5) * 2 * TEMPO_HALF
                    osc_client.send_message("/tempo/raw", float(bpm))

                send_spatial(p["track"], q_rel)

            # Buttons → MIDI chord
            for i in range(4):
                is_pressed = (buttons[i] == 0)
                note_pitch = ROOT_NOTE + intervals[i]
                if is_pressed and not active_note_states[i]:
                    midi_out.send(mido.Message('note_on', channel=0, note=note_pitch, velocity=100))
                    active_note_states[i] = True
                elif not is_pressed and active_note_states[i]:
                    midi_out.send(mido.Message('note_off', channel=0, note=note_pitch, velocity=0))
                    active_note_states[i] = False

            names = " + ".join(PRESETS[p]["name"] for p in sorted(active))
            print(f"[{names}] r={roll:.0f}° p={pitch:.0f}° y={yaw:.0f}° "
                  f"wr={win_roll.center:.0f} wp={win_pitch.center:.0f} wy={win_yaw.center:.0f}   ",
                  end="\r", flush=True)

        except Exception:
            pass

    async with BleakClient(device) as ble:
        await ble.start_notify(GLOVE_CHR, on_notify)
        print(f"\nMatrix active.\n", flush=True)
        await asyncio.Event().wait()


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(f"Usage: presets.py <preset_id> [preset_id ...]   available: {sorted(PRESETS)}")
        sys.exit(1)
    try:
        active = {int(a) for a in args}
    except ValueError:
        print("Preset IDs must be integers.")
        sys.exit(1)
    unknown = active - set(PRESETS)
    if unknown:
        print(f"Unknown preset(s): {unknown}. Available: {sorted(PRESETS)}")
        sys.exit(1)

    print(f"Loading: {' + '.join(PRESETS[p]['name'] for p in sorted(active))}")

    try:
        asyncio.run(main(active))
    except KeyboardInterrupt:
        print("\nHalting. Clearing stuck notes...")
        try:
            midi_out = mido.open_output(IAC_PORT_NAME)
            for i, is_active in enumerate(active_note_states):
                if is_active:
                    midi_out.send(mido.Message('note_off', channel=0,
                                               note=ROOT_NOTE + intervals[i], velocity=0))
        except Exception:
            pass
        print("Done.")
        sys.exit(0)
