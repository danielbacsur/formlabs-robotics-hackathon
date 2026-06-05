import asyncio
import sys

import numpy as np

from formlabs import CHR, PKT, connect, save_cal, to_xyz


STATIC_DURATION = 60.0
MAG_DURATION = 60.0


async def static_calibration():
    """Capture gyro bias AND accel bias from the same flat-and-still period.
    Accel bias assumes gravity reads as [0, 0, +1g] in the body frame when the
    board is flat with components facing up."""
    gyro_samples = []
    acc_samples = []

    def on_notify(_, data):
        gx, gy, gz, ax, ay, az, *_ = PKT.unpack(data)
        gyro_samples.append((gx, gy, gz))
        acc_samples.append((ax, ay, az))

    async with await connect() as ble:
        print(f"STATIC: place the board STILL and FLAT (components up) for {int(STATIC_DURATION)}s ...")
        await ble.start_notify(CHR, on_notify)
        await asyncio.sleep(STATIC_DURATION)
        await ble.stop_notify(CHR)

    gbias = np.mean(gyro_samples, axis=0)
    abias = np.mean(acc_samples, axis=0) - np.array([0.0, 0.0, 1.0])
    print(f"  collected {len(gyro_samples)} samples")
    print(f"  gyro bias = {gbias.tolist()}")
    print(f"  accel bias = {abias.tolist()}")
    save_cal({
        "gyroscope":     {"bias": to_xyz(gbias)},
        "accelerometer": {"bias": to_xyz(abias)},
    })


async def mag_calibration():
    samples = []

    def on_notify(_, data):
        _gx, _gy, _gz, _ax, _ay, _az, mx, my, mz, *_ = PKT.unpack(data)
        samples.append((mx, my, mz))

    async with await connect() as ble:
        print(f"\nMAG: rotate the board through every orientation (figure-8s) for {int(MAG_DURATION)}s ...")
        await ble.start_notify(CHR, on_notify)
        await asyncio.sleep(MAG_DURATION)
        await ble.stop_notify(CHR)

    arr = np.array(samples)
    mn, mx_ = arr.min(axis=0), arr.max(axis=0)
    offset = (mx_ + mn) / 2
    rng = mx_ - mn
    scale = rng.mean() / rng
    print(f"  collected {len(samples)} samples")
    print(f"  min     = {mn.tolist()}")
    print(f"  max     = {mx_.tolist()}")
    print(f"  range   = {rng.tolist()}")
    print(f"  offset  = {offset.tolist()}")
    print(f"  scale   = {scale.tolist()}")
    if (rng < rng.mean() * 0.5).any():
        print("  warning: one axis has <50% range of the others — recapture with more rotation",
              file=sys.stderr)
    save_cal({"magnetometer": {"offset": to_xyz(offset), "scale": to_xyz(scale)}})


async def main():
    await static_calibration()
    print("\nSTATIC done. Pick up the board — mag calibration starts in 5s ...")
    await asyncio.sleep(5)
    await mag_calibration()
    print("\ncalibration complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
