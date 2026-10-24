"""SYNTHETIC data used ONLY to test that the code runs. It is random, has no real-world meaning, and must never be
used for results, figures or the report."""
import cv2
import numpy as np
import pandas as pd


def synthetic_accidents(n=3000, seed=0):
    r = np.random.default_rng(seed)
    hour = r.integers(0, 24, n); d = pd.to_datetime("2022-01-01") + pd.to_timedelta(r.integers(0, 365, n), unit="D")
    speed = r.choice([20, 30, 40, 50, 60, 70], n)
    p = 0.08 + 0.002 * (speed - 20) + 0.05 * ((hour < 5) | (hour > 22))
    sev = np.where(r.random(n) < p, r.choice([1, 2], n), 3)
    df = pd.DataFrame({
        "accident_index": [f"A{i}" for i in range(n)], "latitude": 51 + r.random(n), "longitude": -1 + r.random(n),
        "police_force": r.integers(1, 40, n), "accident_severity": sev, "number_of_vehicles": r.poisson(1.0, n) + 1,
        "number_of_casualties": r.poisson(0.5, n) + 1, "date": d.strftime("%d/%m/%Y"), "day_of_week": ((d.weekday + 1) % 7) + 1,
        "time": [f"{h:02d}:{m:02d}" for h, m in zip(hour, r.integers(0, 60, n))], "speed_limit": speed,
        "road_type": r.choice([1, 2, 3, 6, 9], n), "light_conditions": r.choice([1, 4, 5, 6, 7], n),
        "weather_conditions": r.choice([1, 2, 3, 8, 9, -1], n), "road_surface_conditions": r.choice([1, 2, 3, -1], n),
        "urban_or_rural_area": r.choice([1, 2], n), "junction_detail": r.choice([0, 1, 3, 6, -1], n),
        "did_police_officer_attend_scene_of_accident": r.choice([1, 2], n)})
    df["weather_conditions"] = df["weather_conditions"].astype(object)
    df.loc[5, "weather_conditions"] = "Data missing or out of range"
    return pd.concat([df, df.iloc[:20]], ignore_index=True)  # 20 duplicate rows


def make_scene(seed, size=240):
    r = np.random.default_rng(seed)
    img = (r.random((size, size, 3)) * 60 + 80).astype(np.uint8)
    for _ in range(25):
        x, y, s = r.integers(10, size - 40, 3)
        cv2.rectangle(img, (int(x), int(y)), (int(x + s % 30 + 10), int(y + s % 25 + 10)), tuple(int(v) for v in r.integers(0, 255, 3)), -1)
        cv2.circle(img, (int(y), int(x)), int(s % 15 + 4), tuple(int(v) for v in r.integers(0, 255, 3)), 2)
    return cv2.GaussianBlur(img, (3, 3), 0)


def write_images(root, per_class=30):
    for cls, seed0 in (("classA", 100), ("classB", 500)):
        d = root / cls; d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            r = np.random.default_rng(seed0 + i); img = (r.random((200, 200, 3)) * 50 + 100).astype(np.uint8)
            for _ in range(15):
                x, y = (int(v) for v in r.integers(10, 170, 2))
                if cls == "classA": cv2.circle(img, (x, y), int(r.integers(5, 15)), (255, 255, 255), 2)
                else: cv2.rectangle(img, (x, y), (x + int(r.integers(10, 25)), y + int(r.integers(10, 25))), (0, 0, 0), 2)
            cv2.imwrite(str(d / f"{cls}_{i:03d}.png"), img)


def write_pair(folder):
    folder.mkdir(parents=True, exist_ok=True)
    a = make_scene(1); h, w = a.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 8, 0.95)
    b = cv2.convertScaleAbs(cv2.warpAffine(a, M, (w, h)), alpha=1.1, beta=10)
    cv2.imwrite(str(folder / "scene_a.png"), a); cv2.imwrite(str(folder / "scene_b.png"), b)
