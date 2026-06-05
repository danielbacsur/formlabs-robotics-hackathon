import pickle
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).parent / ".formlabs" / "training.pkl"
MODEL_PATH = Path(__file__).parent / ".formlabs" / "model.pkl"

WINDOW = 50
CLASSES = ["nothing", "left", "right"]

GYRO_PEAK_DPS = 80.0          # local max in |gyro| above this counts as a motion peak
PEAK_SEPARATION_S = 0.15      # minimum gap between two distinct peaks
PAIR_BEFORE_S = 0.5           # peak must be at most this far before keypress
PAIR_AFTER_S = 0.1            # ... or this far after (in case user pressed early)


def find_motion_peaks(times, data):
    gyro_dps = np.linalg.norm(data[:, :3], axis=1)
    above = gyro_dps > GYRO_PEAK_DPS
    peaks = []
    for i in range(1, len(times) - 1):
        if not above[i]:
            continue
        if gyro_dps[i] <= gyro_dps[i - 1] or gyro_dps[i] < gyro_dps[i + 1]:
            continue
        if peaks and times[i] - times[peaks[-1]] < PEAK_SEPARATION_S:
            if gyro_dps[i] > gyro_dps[peaks[-1]]:
                peaks[-1] = i
            continue
        peaks.append(i)
    return peaks


def windows_from_session(samples, events):
    times = np.array([s[0] for s in samples])
    data = np.array([s[1] for s in samples], dtype=np.float32)

    peaks = find_motion_peaks(times, data)
    peak_times = times[peaks]

    # Pair each keypress to its nearest peak within the asymmetric window.
    paired_peak_idx = {}  # peak index -> label
    unpaired_events = 0
    for event_time, label in events:
        lo = event_time - PAIR_BEFORE_S
        hi = event_time + PAIR_AFTER_S
        mask = (peak_times >= lo) & (peak_times <= hi)
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            unpaired_events += 1
            continue
        # nearest peak to keypress, preferring those that came earlier
        nearest = candidates[np.argmin(np.abs(peak_times[candidates] - event_time))]
        peak_idx = peaks[nearest]
        if peak_idx in paired_peak_idx:
            unpaired_events += 1
            continue
        paired_peak_idx[peak_idx] = label

    X, y = [], []
    for peak_idx in peaks:
        if peak_idx < WINDOW - 1:
            continue
        label = paired_peak_idx.get(peak_idx, "nothing")
        X.append(data[peak_idx - WINDOW + 1 : peak_idx + 1])
        y.append(CLASSES.index(label))

    return X, y, len(peaks), len(paired_peak_idx), unpaired_events


def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found — run record.py first")
    with open(DATA_PATH, "rb") as f:
        sessions = pickle.load(f)

    X_all, y_all = [], []
    total_peaks = total_paired = total_unpaired = 0
    for i, s in enumerate(sessions):
        X, y, npks, npair, nun = windows_from_session(s["samples"], s["events"])
        X_all.extend(X)
        y_all.extend(y)
        total_peaks += npks
        total_paired += npair
        total_unpaired += nun
        print(f"  session {i}: {len(s['events'])} events, {npks} motion peaks, "
              f"{npair} paired, {nun} events without nearby peak")

    if total_paired == 0:
        raise SystemExit("no events could be paired with motion peaks — "
                         "lower GYRO_PEAK_DPS or record stronger snaps")

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all)
    counts = {CLASSES[i]: int(c) for i, c in enumerate(np.bincount(y, minlength=len(CLASSES)))}
    print(f"\n{len(sessions)} session(s) → {len(X)} peak-aligned windows  classes={counts}")
    print(f"  {total_paired} hit motions, {total_peaks - total_paired} non-hit motions, "
          f"{total_unpaired} unpaired keypresses (dropped)")

    if (np.bincount(y, minlength=len(CLASSES)) < 5).any():
        print("warning: very few examples in at least one class — record more data")

    X_flat = X.reshape(len(X), -1)
    scaler = StandardScaler().fit(X_flat)
    X_scaled = scaler.transform(X_flat)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=400,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=42,
    )
    model.fit(X_tr, y_tr)

    print(f"\ntrain accuracy: {model.score(X_tr, y_tr):.3f}")
    print(f"test accuracy:  {model.score(X_te, y_te):.3f}")
    cm = confusion_matrix(y_te, model.predict(X_te), labels=list(range(len(CLASSES))))
    print("confusion (rows=true, cols=pred):")
    print("              " + "  ".join(f"{c:>7}" for c in CLASSES))
    for i, row in enumerate(cm):
        print(f"  {CLASSES[i]:<10}" + "  ".join(f"{v:>7}" for v in row))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "scaler": scaler, "window": WINDOW, "classes": CLASSES},
        MODEL_PATH,
    )
    print(f"\nsaved → {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
