import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from formlabs import CHR, PKT, SIDES, connect


REC_DIR = Path(__file__).parent / ".formlabs" / "recordings"


async def record_one(side, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = [0]
    t0 = time.perf_counter()
    f = path.open("w", buffering=1)

    def on_notify(_, data):
        gx, gy, gz, ax, ay, az, mx, my, mz, *buttons = PKT.unpack(data)
        f.write(json.dumps({
            "t": time.perf_counter() - t0,
            "g": [gx, gy, gz],
            "a": [ax, ay, az],
            "m": [mx, my, mz],
            "b": list(buttons),
        }) + "\n")
        count[0] += 1

    try:
        async with await connect(side) as ble:
            print(f"[{side}] recording to {path} — Ctrl-C to stop")
            await ble.start_notify(CHR, on_notify)
            await asyncio.Event().wait()
    finally:
        f.close()
        print(f"[{side}] saved {count[0]} samples to {path}")


async def record(*sides):
    if not sides:
        sides = SIDES
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    await asyncio.gather(*(record_one(s, REC_DIR / f"{stamp}-{s}.jsonl") for s in sides))


if __name__ == "__main__":
    args = sys.argv[1:]
    for a in args:
        if a not in SIDES:
            raise SystemExit(f"usage: record.py [{' | '.join(SIDES)}] ...")
    try:
        asyncio.run(record(*args))
    except KeyboardInterrupt:
        pass
