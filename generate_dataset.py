"""
generate_dataset.py
-------------------
Builds a SYNTHETIC sensor dataset for the miner safety helmet project.

This is NOT real data. It exists so the ML pipeline (preprocessing, training,
evaluation, TFLite conversion) can be developed and tested before a real
helmet is worn by anyone. Replace it with logged ESP32 readings as soon as
you have them.

Units
  Acc*   : g       (helmet at rest -> roughly 1 g spread over the axes)
  Gyro*  : deg/s
  HR     : beats per minute
  SpO2   : percent
  Temp   : degrees C
  Hum    : percent relative humidity

How the classes are built
  NORMAL             walking / working / standing still, vitals and climate fine.
                     Some of them deliberately look a bit "off" (hard work raises
                     heart rate, a hot gallery pushes temperature up) so they
                     overlap with the alert classes.
  FALL               one of two snapshots of a fall:
                       - impact   : big acceleration spike + violent rotation
                       - post-fall: almost no movement, helmet tilted so gravity
                                    is no longer mostly on Z
                     (a single row can't hold "impact then stillness", so both
                     phases are sampled and both are labelled FALL)
  HEALTH_ALERT       high or low heart rate and/or low SpO2, normal movement
  ENVIRONMENT_ALERT  high temperature and/or high humidity, normal movement

Messiness added on purpose
  - overlapping distributions between classes
  - a few mixed cases (e.g. fall + racing heart rate -> labelled FALL)
  - ~1.5 % of labels flipped to a neighbouring class (label noise)
  - a handful of missing readings and sensor glitches so the cleaning
    code in preprocess.py has real work to do
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_SAMPLES = 8000
CLASS_SHARE = {            # roughly what you'd expect: mostly normal operation
    "NORMAL": 0.55,
    "FALL": 0.15,
    "HEALTH_ALERT": 0.15,
    "ENVIRONMENT_ALERT": 0.15,
}
FEATURES = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ",
            "HeartRate", "SpO2", "Temperature", "Humidity"]

rng = np.random.default_rng(SEED)


def random_gravity_split(tilt_deg_sd):
    """Gravity vector (1 g) tilted by a random angle -> (ax, ay, az)."""
    pitch = np.deg2rad(rng.normal(0, tilt_deg_sd))
    roll = np.deg2rad(rng.normal(0, tilt_deg_sd))
    ax = -np.sin(pitch)
    ay = np.sin(roll) * np.cos(pitch)
    az = np.cos(roll) * np.cos(pitch)
    return ax, ay, az


def normal_motion():
    """Walking, working, standing. Helmet mostly upright."""
    activity = rng.choice(["still", "walk", "work", "jolt"], p=[0.25, 0.45, 0.22, 0.08])
    tilt = {"still": 8, "walk": 14, "work": 22, "jolt": 18}[activity]
    acc_noise = {"still": 0.03, "walk": 0.18, "work": 0.30, "jolt": 0.55}[activity]
    gyro_sd = {"still": 4, "walk": 22, "work": 40, "jolt": 85}[activity]
    ax, ay, az = random_gravity_split(tilt)
    acc = np.array([ax, ay, az]) + rng.normal(0, acc_noise, 3)
    gyro = rng.normal(0, gyro_sd, 3)
    return np.concatenate([acc, gyro])


def fall_motion():
    """Either the impact moment or the lying-still moment after it."""
    phase = rng.choice(["impact", "post"], p=[0.55, 0.45])
    if phase == "impact":
        direction = rng.normal(0, 1, 3)
        direction /= np.linalg.norm(direction)
        magnitude = rng.gamma(shape=6.0, scale=0.55) + 1.6      # mostly 2.5 - 6 g
        acc = direction * magnitude + rng.normal(0, 0.35, 3)
        gyro = rng.normal(0, 190, 3)                             # tumbling
    else:
        # lying on the side / back: gravity mostly NOT on Z any more
        angle = np.deg2rad(rng.uniform(35, 100))
        sign = rng.choice([-1, 1])
        acc = np.array([sign * np.sin(angle), rng.normal(0, 0.25), np.cos(angle)])
        acc = acc + rng.normal(0, 0.04, 3)
        gyro = rng.normal(0, 5, 3)
    return np.concatenate([acc, gyro])


def normal_vitals(stress=0.0):
    hr = rng.normal(80 + 25 * stress, 11)
    spo2 = rng.normal(97.3, 1.1)
    return hr, spo2


def normal_climate():
    temp = rng.normal(27.5, 3.2)
    hum = rng.normal(60, 9)
    return temp, hum


def make_row(label):
    if label == "NORMAL":
        motion = normal_motion()
        # about one in six normal miners is working hard / in a warm section
        hard_work = rng.random() < 0.17
        hr, spo2 = normal_vitals(stress=rng.uniform(0.3, 1.0) if hard_work else 0)
        temp, hum = normal_climate()
        if rng.random() < 0.12:
            temp += rng.uniform(2, 6)
        if rng.random() < 0.10:
            hum += rng.uniform(5, 15)

    elif label == "FALL":
        motion = fall_motion()
        hr, spo2 = normal_vitals(stress=rng.uniform(0, 0.9))   # adrenaline
        temp, hum = normal_climate()
        if rng.random() < 0.08:                                # fall that also has bad vitals
            hr = rng.choice([rng.normal(135, 12), rng.normal(48, 5)])
        if rng.random() < 0.18:                                # soft landing, weaker signal
            motion[:3] *= rng.uniform(0.55, 0.8)

    elif label == "HEALTH_ALERT":
        motion = normal_motion()
        if rng.random() < 0.15:                                # collapsed slowly, less movement
            motion[3:] *= 0.3
        kind = rng.choice(["tachy", "brady", "hypoxia", "both"], p=[0.30, 0.20, 0.30, 0.20])
        hr, spo2 = normal_vitals()
        if kind in ("tachy", "both"):
            hr = rng.normal(132, 16)
        if kind == "brady":
            hr = rng.normal(46, 6)
        if kind in ("hypoxia", "both"):
            spo2 = rng.normal(88.5, 3.2)
        temp, hum = normal_climate()

    else:  # ENVIRONMENT_ALERT
        motion = normal_motion()
        hr, spo2 = normal_vitals(stress=rng.uniform(0, 0.5))   # heat raises HR a bit
        kind = rng.choice(["hot", "humid", "both"], p=[0.40, 0.30, 0.30])
        temp, hum = normal_climate()
        if kind in ("hot", "both"):
            temp = rng.normal(41, 4)
        if kind in ("humid", "both"):
            hum = rng.normal(91, 5)

    row = list(motion) + [hr, spo2, temp, hum]
    return row


def main():
    labels = rng.choice(list(CLASS_SHARE), size=N_SAMPLES, p=list(CLASS_SHARE.values()))
    rows = [make_row(lab) for lab in labels]
    df = pd.DataFrame(rows, columns=FEATURES)

    # measurement noise on top of everything (cheap sensors)
    df[["AccX", "AccY", "AccZ"]] += rng.normal(0, 0.02, (N_SAMPLES, 3))
    df[["GyroX", "GyroY", "GyroZ"]] += rng.normal(0, 1.5, (N_SAMPLES, 3))
    df["HeartRate"] += rng.normal(0, 2.0, N_SAMPLES)
    df["SpO2"] += rng.normal(0, 0.4, N_SAMPLES)
    df["Temperature"] += rng.normal(0, 0.4, N_SAMPLES)
    df["Humidity"] += rng.normal(0, 1.5, N_SAMPLES)


    # sensors saturate at their physical range (MPU6050 set to +-8g / +-500 dps)
    df["AccX"] = df["AccX"].clip(-8, 8)
    df["AccY"] = df["AccY"].clip(-8, 8)
    df["AccZ"] = df["AccZ"].clip(-8, 8)
    for g in ("GyroX", "GyroY", "GyroZ"):
        df[g] = df[g].clip(-500, 500)
    df["SpO2"] = df["SpO2"].clip(70, 100)
    df["Humidity"] = df["Humidity"].clip(5, 100)

    df["Label"] = labels

    # label noise: a small share of rows get a wrong-but-plausible label
    confusable = {
        "NORMAL": ["HEALTH_ALERT", "ENVIRONMENT_ALERT"],
        "FALL": ["NORMAL"],
        "HEALTH_ALERT": ["NORMAL"],
        "ENVIRONMENT_ALERT": ["NORMAL"],
    }
    flip = rng.random(N_SAMPLES) < 0.015
    for i in np.where(flip)[0]:
        df.at[i, "Label"] = rng.choice(confusable[df.at[i, "Label"]])

    df = df.round({"AccX": 3, "AccY": 3, "AccZ": 3, "GyroX": 2, "GyroY": 2, "GyroZ": 2,
                   "HeartRate": 1, "SpO2": 1, "Temperature": 2, "Humidity": 1})

    # a few missing readings (sensor dropouts) and glitch values (loose wire, bad contact)
    for col, frac in [("HeartRate", 0.006), ("SpO2", 0.006), ("Temperature", 0.003),
                      ("Humidity", 0.003), ("AccZ", 0.002)]:
        idx = rng.choice(N_SAMPLES, int(N_SAMPLES * frac), replace=False)
        df.loc[idx, col] = np.nan
    idx = rng.choice(N_SAMPLES, 12, replace=False)
    df.loc[idx, "HeartRate"] = rng.choice([0, 255, -1], 12)    # MAX30102 no-finger / overflow
    idx = rng.choice(N_SAMPLES, 8, replace=False)
    df.loc[idx, "SpO2"] = rng.choice([0, 127], 8)
    idx = rng.choice(N_SAMPLES, 6, replace=False)
    df.loc[idx, "Temperature"] = -127.0                        # DS18B20 disconnected value

    out = Path(__file__).parent / "data" / "helmet_sensor_dataset.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Saved {len(df)} rows -> {out}")
    print("\nSamples per class:")
    print(df["Label"].value_counts().to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())


if __name__ == "__main__":
    main()
