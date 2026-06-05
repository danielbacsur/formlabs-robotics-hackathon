import asyncio
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
from ahrs.filters import Madgwick
from bleak import BleakClient, BleakScanner


SVC = "fa9b1d2c-3e4f-4a5b-9c6d-7e8f9a0b1c2d"
CHR = "fa9b1d2c-3e4f-4a5b-9c6d-7e8f9a0b1c2e"
PKT = struct.Struct("<9f4b")

CAL_PATH = Path(__file__).parent / ".formlabs" / "calibration.json"


def xyz(d):
    return np.array([d["x"], d["y"], d["z"]], dtype=float)


def to_xyz(v):
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _deep_merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_cal():
    if not CAL_PATH.exists():
        raise SystemExit(f"{CAL_PATH} not found — run calibrate.py first")
    return json.loads(CAL_PATH.read_text())


def save_cal(updates):
    CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    cal = json.loads(CAL_PATH.read_text()) if CAL_PATH.exists() else {}
    _deep_merge(cal, updates)
    CAL_PATH.write_text(json.dumps(cal, indent=2))
    print(f"wrote {CAL_PATH}: {updates}")


async def connect():
    device = await BleakScanner.find_device_by_filter(
        lambda _, advert: SVC in (uuid.lower() for uuid in advert.service_uuids),
        timeout=20.0,
    )
    if device is None:
        raise SystemExit(f"no peripheral advertising {SVC}")
    return BleakClient(device)


def quat_to_euler(q):
    # ahrs uses scalar-first [w, x, y, z], NED. Returns roll, pitch, yaw in degrees.
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


async def stream():
    cal = load_cal()
    gbias = xyz(cal["gyroscope"]["bias"])
    abias = xyz(cal["accelerometer"]["bias"])
    moff = xyz(cal["magnetometer"]["offset"])
    mscale = xyz(cal["magnetometer"]["scale"])

    madgwick = Madgwick(gain=0.02)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    last_t = [None]

    def on_notify(_, data):
        gx, gy, gz, ax, ay, az, mx, my, mz, *buttons = PKT.unpack(data)

        raw_gyr = np.array([gx, gy, gz])
        gyr = (raw_gyr - gbias) * (math.pi / 180.0)
        gyr[2] = -gyr[2]  # Rev2 BMI270 yaw axis
        acc = np.array([ax, ay, az]) - abias
        mag = (np.array([mx, my, mz]) - moff) * mscale
        mag[1:] = -mag[1:]  # Rev2 BMM150 axis fix

        now = time.perf_counter()
        dt = 0.01 if last_t[0] is None else max(now - last_t[0], 1e-4)
        last_t[0] = now
        madgwick.Dt = dt

        # Refine gyro bias only when truly still — decoupled from filter dynamics.
        if 0.95 < np.linalg.norm(acc) < 1.05 and np.linalg.norm(gyr) < math.radians(2):
            gbias[:] = 0.998 * gbias + 0.002 * raw_gyr

        q[:] = madgwick.updateMARG(q, gyr=gyr, acc=acc, mag=mag)
        roll, pitch, yaw = quat_to_euler(q)
        print(f"roll={roll} pitch={pitch} yaw={yaw}  buttons={buttons}")

    async with await connect() as ble:
        await ble.start_notify(CHR, on_notify)
        await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(stream())
    except KeyboardInterrupt:
        pass
