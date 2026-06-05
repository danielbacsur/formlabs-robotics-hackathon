import asyncio
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
from ahrs.filters import Madgwick
from bleak import BleakClient, BleakScanner


SVC = "fa9b1d2c-3e4f-4a5b-9c6d-7e8f9a0b1c2d"
CHR = "fa9b1d2c-3e4f-4a5b-9c6d-7e8f9a0b1c2e"
PKT = struct.Struct("<9f4b")

CAL_PATH = Path(__file__).with_name("calibration.json")
DEFAULT_CAL = {
    "gyro_bias": [0.0, 0.0, 0.0],
    "mag_offset": [0.0, 0.0, 0.0],
    "mag_scale": [1.0, 1.0, 1.0],
    "declination_deg": 0.0,
}


def load_cal():
    if not CAL_PATH.exists():
        print(f"warning: {CAL_PATH.name} not found; running uncalibrated", file=sys.stderr)
        return DEFAULT_CAL
    return {**DEFAULT_CAL, **json.loads(CAL_PATH.read_text())}


def save_cal(updates):
    cal = json.loads(CAL_PATH.read_text()) if CAL_PATH.exists() else {}
    cal.update(updates)
    CAL_PATH.write_text(json.dumps(cal, indent=2))
    print(f"wrote {CAL_PATH.name}: {updates}")


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
    gbias = np.array(cal["gyro_bias"], dtype=float)
    moff = np.array(cal["mag_offset"], dtype=float)
    mscale = np.array(cal["mag_scale"], dtype=float)
    decl = float(cal["declination_deg"])

    madgwick = Madgwick(gain=0.02)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    last_t = [None]

    def on_notify(_, data):
        gx, gy, gz, ax, ay, az, mx, my, mz, *btn = PKT.unpack(data)

        raw_gyr = np.array([gx, gy, gz])  # raw deg/s, pre-flip — matches gbias frame
        gyr = (raw_gyr - gbias) * (math.pi / 180.0)
        gyr[2] = -gyr[2]  # Rev2 BMI270 yaw direction matches NED only after Z flip
        acc = np.array([ax, ay, az])
        mag = (np.array([mx, my, mz]) - moff) * mscale
        mag[1] = -mag[1]  # Rev2 BMM150 axis fix (per Reefwing-AHRS)
        mag[2] = -mag[2]

        now = time.perf_counter()
        dt = 0.01 if last_t[0] is None else max(now - last_t[0], 1e-4)
        last_t[0] = now
        madgwick.Dt = dt

        # Continuous bias refinement via per-sample EMA whenever the board is
        # still. Filter dynamics untouched.
        if 0.95 < np.linalg.norm(acc) < 1.05 and np.linalg.norm(gyr) < math.radians(2):
            gbias[:] = 0.998 * gbias + 0.002 * raw_gyr

        q[:] = madgwick.updateMARG(q, gyr=gyr, acc=acc, mag=mag)
        roll, pitch, yaw = quat_to_euler(q)
        heading = (yaw + decl) % 360.0

        print(
            f"roll={roll:7.2f} pitch={pitch:7.2f} yaw={yaw:7.2f} "
            f"heading={heading:6.2f}  btn={btn}"
        )

    async with await connect() as ble:
        await ble.start_notify(CHR, on_notify)
        await asyncio.Event().wait()


async def calibrate_gyro(duration=3.0):
    samples = []

    def on_notify(_, data):
        gx, gy, gz, *_ = PKT.unpack(data)
        samples.append((gx, gy, gz))

    async with await connect() as ble:
        print(f"keep the board STILL for {duration:.0f}s ...")
        await ble.start_notify(CHR, on_notify)
        await asyncio.sleep(duration)
        await ble.stop_notify(CHR)

    bias = np.mean(samples, axis=0).tolist()
    print(f"collected {len(samples)} samples, bias = {bias}")
    save_cal({"gyro_bias": bias})


async def calibrate_mag(duration=30.0):
    samples = []

    def on_notify(_, data):
        _gx, _gy, _gz, _ax, _ay, _az, mx, my, mz, *_ = PKT.unpack(data)
        samples.append((mx, my, mz))

    async with await connect() as ble:
        print(f"rotate the board through every orientation (figure-8s) for {duration:.0f}s ...")
        await ble.start_notify(CHR, on_notify)
        await asyncio.sleep(duration)
        await ble.stop_notify(CHR)

    arr = np.array(samples)
    mn, mx_ = arr.min(axis=0), arr.max(axis=0)
    offset = ((mx_ + mn) / 2).tolist()
    rng = mx_ - mn
    scale = (rng.mean() / rng).tolist()
    print(f"collected {len(samples)} samples")
    print(f"  min     = {mn.tolist()}")
    print(f"  max     = {mx_.tolist()}")
    print(f"  range   = {rng.tolist()}")
    print(f"  offset  = {offset}")
    print(f"  scale   = {scale}")
    if (rng < rng.mean() * 0.5).any():
        print("warning: one axis has <50% range of the others — recapture with more rotation",
              file=sys.stderr)
    save_cal({"mag_offset": offset, "mag_scale": scale})


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def visualize():
    import threading
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    cal = load_cal()
    gbias = np.array(cal["gyro_bias"], dtype=float)
    moff = np.array(cal["mag_offset"], dtype=float)
    mscale = np.array(cal["mag_scale"], dtype=float)
    decl = float(cal["declination_deg"])

    madgwick = Madgwick(gain=0.02)
    state = {"q": np.array([1.0, 0.0, 0.0, 0.0]), "rpyh": (0.0, 0.0, 0.0, 0.0), "still": False}
    last_t = [None]

    def on_notify(_, data):
        gx, gy, gz, ax, ay, az, mx, my, mz, *_ = PKT.unpack(data)
        raw_gyr = np.array([gx, gy, gz])
        gyr = (raw_gyr - gbias) * (math.pi / 180.0)
        gyr[2] = -gyr[2]  # Rev2 BMI270 yaw direction matches NED only after Z flip
        acc = np.array([ax, ay, az])
        mag = (np.array([mx, my, mz]) - moff) * mscale
        mag[1], mag[2] = -mag[1], -mag[2]

        now = time.perf_counter()
        dt = 0.01 if last_t[0] is None else max(now - last_t[0], 1e-4)
        last_t[0] = now
        madgwick.Dt = dt

        still = (0.95 < np.linalg.norm(acc) < 1.05
                 and np.linalg.norm(gyr) < math.radians(2))
        if still:
            gbias[:] = 0.998 * gbias + 0.002 * raw_gyr

        state["still"] = still

        state["q"] = np.asarray(madgwick.updateMARG(state["q"], gyr=gyr, acc=acc, mag=mag))
        r, p, y = quat_to_euler(state["q"])
        state["rpyh"] = (r, p, y, (y + decl) % 360.0)

    def run_ble():
        async def loop():
            async with await connect() as ble:
                await ble.start_notify(CHR, on_notify)
                await asyncio.Event().wait()
        asyncio.run(loop())

    threading.Thread(target=run_ble, daemon=True).start()

    # Nano 33 BLE physical proportions (length x width x height, normalized)
    L, W, H = 0.45, 0.18, 0.02
    verts = np.array([
        [-L/2, -W/2, -H/2], [L/2, -W/2, -H/2], [L/2, W/2, -H/2], [-L/2, W/2, -H/2],
        [-L/2, -W/2,  H/2], [L/2, -W/2,  H/2], [L/2, W/2,  H/2], [-L/2, W/2,  H/2],
    ])
    faces_idx = [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [0,3,7,4], [1,2,6,5]]
    face_colors = ["#1f77b4", "#ff7f0e", "#888", "#888", "#888", "#888"]  # bottom blue, top orange

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
        ax.invert_zaxis()  # NED: +Z is down, visually flip so gravity points down
        ax.invert_xaxis()  # flip X so the red N arrow points the visually opposite way

        # world reference frame
        ax.quiver(0, 0, 0, 0.5, 0, 0, color="red",   linewidth=2)
        ax.quiver(0, 0, 0, 0, 0.5, 0, color="green", linewidth=2)
        ax.quiver(0, 0, 0, 0, 0, 0.5, color="blue",  linewidth=2)
        ax.text(0.55, 0, 0, "N", color="red")
        ax.text(0, 0.55, 0, "E", color="green")
        ax.text(0, 0, 0.55, "D", color="blue")

        # rotated board (body frame -> world frame)
        R = quat_to_rotmat(state["q"])
        rotated = verts @ R.T
        poly = Poly3DCollection(
            [rotated[f] for f in faces_idx],
            facecolors=face_colors, edgecolor="k", alpha=0.7,
        )
        ax.add_collection3d(poly)

        # body +X arrow (USB-opposite edge), shows where the board "points"
        nose = R @ np.array([L * 0.7, 0, 0])
        ax.quiver(0, 0, 0, nose[0], nose[1], nose[2], color="magenta", linewidth=2)

        r, p, y, h = state["rpyh"]
        tag = "  [bias↻]" if state["still"] else ""
        ax.set_title(
            f"roll={r:+6.1f}°  pitch={p:+6.1f}°  yaw={y:+6.1f}°  heading={h:6.1f}°{tag}",
            fontfamily="monospace",
        )

    anim = FuncAnimation(fig, update, interval=33, cache_frame_data=False)
    fig._anim = anim  # keep reference alive
    plt.show()


MODES = {
    "stream": stream,
    "calibrate-gyro": calibrate_gyro,
    "calibrate-mag": calibrate_mag,
    "visualize": visualize,
}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "stream"
    if mode not in MODES:
        raise SystemExit(f"usage: formlabs.py [{'|'.join(MODES)}]")
    fn = MODES[mode]
    try:
        if asyncio.iscoroutinefunction(fn):
            asyncio.run(fn())
        else:
            fn()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
