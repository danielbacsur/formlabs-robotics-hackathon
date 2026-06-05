import asyncio
import pickle
import time
from pathlib import Path

from pynput import keyboard

from formlabs import CHR, PKT, connect


DATA_PATH = Path(__file__).parent / ".formlabs" / "training.pkl"
SIDE = "right"


def main():
    samples = []
    events = []
    stop = [False]

    def on_press(key):
        try:
            c = key.char.lower()
        except AttributeError:
            return
        if c == "a":
            events.append((time.perf_counter(), "left"))
            print("  LEFT", flush=True)
        elif c == "d":
            events.append((time.perf_counter(), "right"))
            print("  RIGHT", flush=True)
        elif c == "q":
            stop[0] = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    def on_notify(_, data):
        t = time.perf_counter()
        samples.append((t, list(PKT.unpack(data)[:9])))

    async def run():
        async with await connect(SIDE) as ble:
            await ble.start_notify(CHR, on_notify)
            print(f"connected to '{SIDE}'")
            print("  press 'a' for LEFT drum, 'd' for RIGHT drum, 'q' to stop & save")
            print("  (keep this terminal focused so keypresses are captured)")
            while not stop[0]:
                await asyncio.sleep(0.1)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()

    if not events:
        print("no labels recorded, nothing saved")
        return

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    sessions = []
    if DATA_PATH.exists():
        with open(DATA_PATH, "rb") as f:
            sessions = pickle.load(f)
    sessions.append({"samples": samples, "events": events})
    with open(DATA_PATH, "wb") as f:
        pickle.dump(sessions, f)

    n_left = sum(1 for _, lbl in events if lbl == "left")
    n_right = sum(1 for _, lbl in events if lbl == "right")
    tot_samples = sum(len(s["samples"]) for s in sessions)
    tot_events = sum(len(s["events"]) for s in sessions)
    print(f"saved → {DATA_PATH.name}")
    print(f"  this session: {len(samples)} samples, {len(events)} events ({n_left} L, {n_right} R)")
    print(f"  cumulative:   {tot_samples} samples, {tot_events} events across {len(sessions)} sessions")


if __name__ == "__main__":
    main()
