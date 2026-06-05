import asyncio
import json
import math
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
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
    """Accel-style roll/pitch + mag-style heading. Each component is read off
    a different geometric projection of the quaternion, so they don't
    cross-couple at extreme angles the way ZYX Euler does:
      - roll: rotation of body+Z away from world+Z around body+X
      - pitch: tilt of body+X above the horizon (+ = nose up)
      - heading: compass bearing of body+X (0=N, 90=E, 180=S, 270=W)"""
    w, x, y, z = q
    # world+Z (up) expressed in body coords = third row of body→world R
    ux = 2 * (x * z - w * y)
    uy = 2 * (y * z + w * x)
    uz = 1 - 2 * (x * x + y * y)
    roll = math.atan2(uy, uz)
    pitch = math.atan2(ux, math.sqrt(uy * uy + uz * uz))
    # body+X (nose) expressed in world coords = first column of R
    fx = 1 - 2 * (y * y + z * z)
    fy = 2 * (x * y + w * z)
    heading = math.atan2(-fy, fx)
    return math.degrees(roll), math.degrees(pitch), math.degrees(heading)


def _quat_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def _quat_integrate(q, omega, dt):
    wx, wy, wz = omega
    w, x, y, z = q
    dq = 0.5 * dt * np.array([
        -x * wx - y * wy - z * wz,
         w * wx + y * wz - z * wy,
         w * wy - x * wz + z * wx,
         w * wz + x * wy - y * wx,
    ])
    return _quat_normalize(q + dq)


def _slerp(q1, q2, t):
    dot = float(np.dot(q1, q2))
    if dot < 0:
        q2, dot = -q2, -dot
    if dot > 0.9995:
        return _quat_normalize(q1 + t * (q2 - q1))
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_0 = math.sin(theta_0)
    theta = theta_0 * t
    s2 = math.sin(theta) / sin_0
    s1 = math.cos(theta) - dot * s2
    return s1 * q1 + s2 * q2


def _rotmat_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return _quat_normalize(np.array([
            0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
        ]))
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return _quat_normalize(np.array([
            (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
        ]))
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return _quat_normalize(np.array([
            (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
        ]))
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return _quat_normalize(np.array([
        (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    ]))


def _quat_from_acc_mag(acc, mag):
    """Body->world (NWU) quaternion from accel + tilt-compensated mag.
    Returns None if either vector is degenerate."""
    a_n = np.linalg.norm(acc)
    m_n = np.linalg.norm(mag)
    if a_n < 1e-6 or m_n < 1e-6:
        return None
    up = acc / a_n
    m = mag / m_n
    m_h = m - up * float(np.dot(m, up))
    nh = np.linalg.norm(m_h)
    if nh < 1e-6:
        return None
    north = m_h / nh
    west = np.cross(up, north)
    return _rotmat_to_quat(np.array([north, west, up]))


class OrientationFilter:
    """Quaternion complementary filter:
      - gyro integrates the prediction
      - accel + tilt-compensated mag give a measurement quaternion
      - SLERP toward the measurement with a time-constant gain that's fast
        when still and slow during motion (rejects linear accel & magnetic
        disturbances)
      - gyro bias is adapted only during sustained still periods"""

    TAU_STILL = 0.3
    TAU_MOVE = 1.0
    STILL_FRAMES = 20

    def __init__(self, cal):
        self.gbias = xyz(cal["gyroscope"]["bias"]).copy()
        self.abias = xyz(cal["accelerometer"]["bias"])
        self.moff = xyz(cal["magnetometer"]["offset"])
        self.mscale = xyz(cal["magnetometer"]["scale"])
        self.q = None
        self._still_n = 0

    def update(self, raw_gyr, raw_acc, raw_mag, dt):
        gyr = (raw_gyr - self.gbias) * (math.pi / 180.0)
        acc = raw_acc - self.abias
        mag = (raw_mag - self.moff) * self.mscale
        # On the Nano 33 BLE Sense the Y axis is reported with a flipped
        # sign on gyro, accel, and mag (the BMI270 gyro and BMM150 mag are
        # mounted upside-down on Y relative to the accelerometer's printed
        # board frame). Bring everything into the same right-handed body
        # frame.
        gyr[1] = -gyr[1]
        acc[1] = -acc[1]
        mag[1] = -mag[1]
        mag[2] = -mag[2]

        if self.q is None:
            q0 = _quat_from_acc_mag(acc, mag)
            self.q = q0 if q0 is not None else np.array([1.0, 0.0, 0.0, 0.0])

        self.q = _quat_integrate(self.q, gyr, dt)
        q_meas = _quat_from_acc_mag(acc, mag)
        still = False
        if q_meas is not None:
            a_n = np.linalg.norm(acc)
            still = abs(a_n - 1.0) < 0.03 and float(np.linalg.norm(gyr)) < math.radians(2)
            tau = self.TAU_STILL if still else self.TAU_MOVE
            self.q = _slerp(self.q, q_meas, 1.0 - math.exp(-dt / tau))

        if still:
            self._still_n += 1
            if self._still_n > self.STILL_FRAMES:
                self.gbias = 0.99 * self.gbias + 0.01 * raw_gyr
        else:
            self._still_n = 0

        return self.q, still


async def stream_one(side):
    flt = OrientationFilter(load_cal(side))
    last_t = [None]

    def on_notify(_, data):
        gx, gy, gz, ax, ay, az, mx, my, mz, *buttons = PKT.unpack(data)
        now = time.perf_counter()
        dt = 0.01 if last_t[0] is None else max(min(now - last_t[0], 0.2), 1e-4)
        last_t[0] = now
        q, _ = flt.update(np.array([gx, gy, gz]), np.array([ax, ay, az]),
                          np.array([mx, my, mz]), dt)
        roll, pitch, heading = quat_to_euler(q)
        print(f"[{side}] roll={roll:+.2f} pitch={pitch:+.2f} heading={heading:+.2f}  buttons={buttons}")

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

    flt = OrientationFilter(load_cal(side))
    state = {"q": np.array([1.0, 0.0, 0.0, 0.0]), "still": False}
    last_t = [None]

    def on_notify(_, data):
        gx, gy, gz, ax, ay, az, mx, my, mz, *_ = PKT.unpack(data)
        now = time.perf_counter()
        dt = 0.01 if last_t[0] is None else max(min(now - last_t[0], 0.2), 1e-4)
        last_t[0] = now
        q, still = flt.update(np.array([gx, gy, gz]), np.array([ax, ay, az]),
                              np.array([mx, my, mz]), dt)
        state["q"] = q
        state["still"] = still

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
        ax.set_ylabel("Y — West")
        ax.set_zlabel("Z — Up")

        ax.quiver(0, 0, 0, 0.5, 0, 0, color="red",   linewidth=2)
        ax.quiver(0, 0, 0, 0, 0.5, 0, color="green", linewidth=2)
        ax.quiver(0, 0, 0, 0, 0, 0.5, color="blue",  linewidth=2)
        ax.text(0.55, 0, 0, "N", color="red")
        ax.text(0, 0.55, 0, "W", color="green")
        ax.text(0, 0, 0.55, "U", color="blue")

        R = quat_to_rotmat(state["q"])
        rotated = verts @ R.T
        poly = Poly3DCollection(
            [rotated[f] for f in faces_idx],
            facecolors=face_colors, edgecolor="k", alpha=0.7,
        )
        ax.add_collection3d(poly)

        nose = R @ np.array([L * 0.7, 0, 0])
        ax.quiver(0, 0, 0, nose[0], nose[1], nose[2], color="magenta", linewidth=2)

        r, p, h = quat_to_euler(state["q"])
        tag = "  [bias↻]" if state["still"] else ""
        ax.set_title(
            f"[{side}]  roll={r:+6.1f}°  pitch={p:+6.1f}°  heading={h:+6.1f}°{tag}",
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
