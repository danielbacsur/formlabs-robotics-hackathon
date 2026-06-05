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
# =====           CONFIG MATRIX        =====
# ==========================================
GLOVE_SIDE = "right"  # "left" or "right"
GLOVE_SVC  = "fa9b1d2c-3e4f-4a5b-9c6d-7e8f9a0b1c2d"
GLOVE_CHR  = "fa9b1d2c-3e4f-4a5b-9c6d-7e8f9a0b1c2e"
GLOVE_PKT  = struct.Struct("<9f4b")
CAL_PATH   = Path(__file__).parent.parent / "python" / "formlabs" / ".formlabs" / "calibration.json"

IAC_PORT_NAME = 'IAC illeszt≈ëprogram 1. busz'

osc_client     = udp_client.SimpleUDPClient("127.0.0.1", 8000)
MASTER_VOL_OSC = "/master/volume"
PITCH_MIN      = -45.0
PITCH_MAX      =  45.0

# Ambisonics rotation plugin — 7th in chain, quaternion params 3-6
FX           = 7
AMB_W        = f"/track/1/fx/{FX}/fxparam/3/value"
AMB_X        = f"/track/1/fx/{FX}/fxparam/4/value"
AMB_Y        = f"/track/1/fx/{FX}/fxparam/5/value"
AMB_Z        = f"/track/1/fx/{FX}/fxparam/6/value"
WARMUP_PKTS  = 50  # packets before locking forward reference (~0.5 s)

ACTIVE_SCALE_MODE = "aeolian"
SCALE_LIBRARY = {
    "aeolian": [0, 3, 7, 10],
    "ionian":  [0, 4, 7, 11],
    "dorian":  [0, 3, 7, 9],
}
ROOT_NOTE          = 60
intervals          = SCALE_LIBRARY.get(ACTIVE_SCALE_MODE, SCALE_LIBRARY["aeolian"])
active_note_states = [False, False, False, False]


# ==========================================
# =====    CALIBRATION / ORIENTATION   =====
# ==========================================

def _xyz(d):
    return np.array([d["x"], d["y"], d["z"]], dtype=float)


def load_cal():
    cal = json.loads(CAL_PATH.read_text())
    return cal[GLOVE_SIDE]


def quat_to_euler(q):
    w, x, y, z = q
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp  = 2*(w*y - z*x)
    pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


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
    return np.array([w, -x, -y, -z])  # conjugate = inverse for unit quaternions


def send_quat_osc(q):
    w, x, y, z = q
    osc_client.send_message(AMB_W, float((w + 1) / 2))
    osc_client.send_message(AMB_X, float((x + 1) / 2))
    osc_client.send_message(AMB_Y, float((y + 1) / 2))
    osc_client.send_message(AMB_Z, float((-z + 1) / 2))  # inverted for left-right


# ==========================================
# =====             MAIN               =====
# ==========================================

async def main():
    global active_note_states

    cal    = load_cal()
    gbias  = _xyz(cal["gyroscope"]["bias"])
    abias  = _xyz(cal["accelerometer"]["bias"])
    moff   = _xyz(cal["magnetometer"]["offset"])
    mscale = _xyz(cal["magnetometer"]["scale"])

    madgwick = Madgwick(gain=0.02)
    q        = np.array([1.0, 0.0, 0.0, 0.0])
    last_t   = [None]
    smoothed = [0.5]
    warmup   = [0]
    q_ref    = [None]

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
        print(f"SUCCESS: Bound to virtual MIDI pipeline -> {IAC_PORT_NAME}")
    except OSError:
        print(f"FATAL: Could not find virtual port '{IAC_PORT_NAME}'. Check CoreMIDI settings.")
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
            roll, pitch, yaw = quat_to_euler(q)

            # Wait for Madgwick to converge, then lock forward reference
            warmup[0] += 1
            if warmup[0] < WARMUP_PKTS:
                return
            if q_ref[0] is None:
                q_ref[0] = q.copy()
                send_quat_osc(np.array([1.0, 0.0, 0.0, 0.0]))  # zero the plugin
                print("\nForward reference locked. Ambisonics zeroed.", flush=True)
                return

            # Ambisonics: relative rotation from forward reference
            q_rel = q_mul(q_inv(q_ref[0]), q)
            send_quat_osc(q_rel)

            # MIDI chord from buttons
            for i in range(4):
                is_pressed = (buttons[i] == 0)
                note_pitch = ROOT_NOTE + intervals[i]
                if is_pressed and not active_note_states[i]:
                    midi_out.send(mido.Message('note_on', channel=0, note=note_pitch, velocity=100))
                    active_note_states[i] = True
                elif not is_pressed and active_note_states[i]:
                    midi_out.send(mido.Message('note_off', channel=0, note=note_pitch, velocity=0))
                    active_note_states[i] = False

            print(f"[{GLOVE_SIDE}] roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f} buttons={list(buttons)} vol={smoothed[0]:.2f}   ", end="\r", flush=True)

        except Exception as e:
            pass

    async with BleakClient(device) as ble:
        await ble.start_notify(GLOVE_CHR, on_notify)
        print(f"\nMatrix fully active. Hold still for {WARMUP_PKTS} packets to zero...\n", flush=True)
        await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nHalting. Clearing stuck notes...")
        try:
            midi_out = mido.open_output(IAC_PORT_NAME)
            for i, is_active in enumerate(active_note_states):
                note_pitch = ROOT_NOTE + intervals[i]
                midi_out.send(mido.Message('note_off', channel=0, note=note_pitch, velocity=0))
        except Exception:
            pass
        print("System safely offline.")
        sys.exit(0)
