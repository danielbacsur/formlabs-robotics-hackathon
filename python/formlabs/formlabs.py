import asyncio
import json
import math
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
from ahrs.filters import Madgwick
from bleak import BleakClient, BleakScanner


SVC = "fa9b1d2c-3e4f-4a5b-9c6d-7e8f9a0b1c2d"
CHR = "fa9b1d2c-3e4f-4a5b-9c6d-7e8f9a0b1c2e"
PKT = struct.Struct("<9f4b")

SIDES = ("left", "right")
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


def load_cal(side):
    if not CAL_PATH.exists():
        raise SystemExit(f"{CAL_PATH} not found — run calibration.py {side} first")
    cal = json.loads(CAL_PATH.read_text())
    if side not in cal:
        raise SystemExit(f"no calibration for '{side}' — run calibration.py {side}")
    return cal[side]


def save_cal(side, updates):
    CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    cal = json.loads(CAL_PATH.read_text()) if CAL_PATH.exists() else {}
    cal.setdefault(side, {})
    _deep_merge(cal[side], updates)
    CAL_PATH.write_text(json.dumps(cal, indent=2))
    print(f"wrote {CAL_PATH} [{side}]: {updates}")


async def connect(side):
    device = await BleakScanner.find_device_by_filter(
        lambda _, advert: (
            SVC in (u.lower() for u in advert.service_uuids)
            and (advert.local_name or "").lower() == side
        ),
        timeout=20.0,
    )
    if device is None:
        raise SystemExit(f"no '{side}' peripheral advertising {SVC}")
    return BleakClient(device)


def quat_to_euler(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


async def stream_one(side):
    cal = load_cal(side)
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
        gyr[2] = -gyr[2]
        acc = np.array([ax, ay, az]) - abias
        mag = (np.array([mx, my, mz]) - moff) * mscale
        mag[1:] = -mag[1:]

        now = time.perf_counter()
        dt = 0.01 if last_t[0] is None else max(now - last_t[0], 1e-4)
        last_t[0] = now
        madgwick.Dt = dt

        if 0.95 < np.linalg.norm(acc) < 1.05 and np.linalg.norm(gyr) < math.radians(2):
            gbias[:] = 0.998 * gbias + 0.002 * raw_gyr

        q[:] = madgwick.updateMARG(q, gyr=gyr, acc=acc, mag=mag)
        roll, pitch, yaw = quat_to_euler(q)
        print(f"[{side}] roll={roll} pitch={pitch} yaw={yaw}  buttons={buttons}")

    async with await connect(side) as ble:
        await ble.start_notify(CHR, on_notify)
        await asyncio.Event().wait()


async def stream(*sides):
    if not sides:
        sides = SIDES
    await asyncio.gather(*(stream_one(s) for s in sides))


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def visualize(side):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    cal = load_cal(side)
    gbias = xyz(cal["gyroscope"]["bias"])
    abias = xyz(cal["accelerometer"]["bias"])
    moff = xyz(cal["magnetometer"]["offset"])
    mscale = xyz(cal["magnetometer"]["scale"])

    madgwick = Madgwick(gain=0.02)
    state = {"q": np.array([1.0, 0.0, 0.0, 0.0]), "still": False}
    last_t = [None]

    def on_notify(_, data):
        gx, gy, gz, ax, ay, az, mx, my, mz, *_ = PKT.unpack(data)
        raw_gyr = np.array([gx, gy, gz])
        gyr = (raw_gyr - gbias) * (math.pi / 180.0)
        gyr[2] = -gyr[2]
        acc = np.array([ax, ay, az]) - abias
        mag = (np.array([mx, my, mz]) - moff) * mscale
        mag[1:] = -mag[1:]

        now = time.perf_counter()
        dt = 0.01 if last_t[0] is None else max(now - last_t[0], 1e-4)
        last_t[0] = now
        madgwick.Dt = dt

        still = 0.95 < np.linalg.norm(acc) < 1.05 and np.linalg.norm(gyr) < math.radians(2)
        if still:
            gbias[:] = 0.998 * gbias + 0.002 * raw_gyr
        state["still"] = still

        state["q"] = np.asarray(madgwick.updateMARG(state["q"], gyr=gyr, acc=acc, mag=mag))

    def run_ble():
        async def loop():
            async with await connect(side) as ble:
                await ble.start_notify(CHR, on_notify)
                await asyncio.Event().wait()
        asyncio.run(loop())

    threading.Thread(target=run_ble, daemon=True).start()

    L, W, H = 0.45, 0.18, 0.02
    verts = np.array([
        [-L/2, -W/2, -H/2], [L/2, -W/2, -H/2], [L/2, W/2, -H/2], [-L/2, W/2, -H/2],
        [-L/2, -W/2,  H/2], [L/2, -W/2,  H/2], [L/2, W/2,  H/2], [-L/2, W/2,  H/2],
    ])
    faces_idx = [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [0,3,7,4], [1,2,6,5]]
    face_colors = ["#1f77b4", "#ff7f0e", "#888", "#888", "#888", "#888"]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    def update(_frame):
        ax.clear()
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(-0.6, 0.6)
        ax.set_xlabel("X — North")
        ax.set_ylabel("Y — East")
        ax.set_zlabel("Z — Down")
        ax.invert_zaxis()
        ax.invert_xaxis()

        ax.quiver(0, 0, 0, 0.5, 0, 0, color="red",   linewidth=2)
        ax.quiver(0, 0, 0, 0, 0.5, 0, color="green", linewidth=2)
        ax.quiver(0, 0, 0, 0, 0, 0.5, color="blue",  linewidth=2)
        ax.text(0.55, 0, 0, "N", color="red")
        ax.text(0, 0.55, 0, "E", color="green")
        ax.text(0, 0, 0.55, "D", color="blue")

        R = quat_to_rotmat(state["q"])
        rotated = verts @ R.T
        poly = Poly3DCollection(
            [rotated[f] for f in faces_idx],
            facecolors=face_colors, edgecolor="k", alpha=0.7,
        )
        ax.add_collection3d(poly)

        nose = R @ np.array([L * 0.7, 0, 0])
        ax.quiver(0, 0, 0, nose[0], nose[1], nose[2], color="magenta", linewidth=2)

        r, p, y = quat_to_euler(state["q"])
        tag = "  [bias↻]" if state["still"] else ""
        ax.set_title(
            f"[{side}]  roll={r:+6.1f}°  pitch={p:+6.1f}°  yaw={y:+6.1f}°{tag}",
            fontfamily="monospace",
        )

    anim = FuncAnimation(fig, update, interval=33, cache_frame_data=False)
    fig._anim = anim
    plt.show()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "visualize":
        if len(args) != 2 or args[1] not in SIDES:
            raise SystemExit(f"usage: formlabs.py visualize [{' | '.join(SIDES)}]")
        visualize(args[1])
    else:
        for a in args:
            if a not in SIDES:
                raise SystemExit(
                    f"usage: formlabs.py [{' | '.join(SIDES)}] ...\n"
                    f"       formlabs.py visualize [{' | '.join(SIDES)}]"
                )
        try:
            asyncio.run(stream(*args))
        except KeyboardInterrupt:
            pass
