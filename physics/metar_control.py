"""Score the icing physics against routine weather observations (review
2026-09-14, comment 7, the experiment Arno rates highest).

The icing evaluation of validate_carb_icing.py contrasts accidents with
accidents, so it cannot show that the physics separates icing weather from
the weather of uneventful flying. Routine hourly METAR observations at the
same airports supply that population. For every accident in the study set
the script locates the nearest NOAA ISD station to the recorded coordinates
(preferring the observing facility the record names), downloads the
station-year of ISD-Lite hourly observations, and draws the routine hours of
the accident's calendar month at that station as the control. Two control
constructions are reported: every hour of the station-month, and the hours
within three hours of the accident's clock time on every day of the month,
which matches the diurnal cycle of the exposure.

Study set: every weather-recorded NTSB accident whose chain carries
carburetor icing (the 1,205 positives of the paper), plus a seeded random
sample of other weather-recorded accidents as the reference.

Reported:
  * ROC area of the chart icing probability (descent power) for icing
    accidents against routine hours, and for the other accidents against
    routine hours, with DeLong intervals;
  * icing accidents against the other accidents on the same subset, for
    comparison with the 0.612 of the accident-only evaluation;
  * exposure-normalized relative risk by chart zone: the share of icing
    accidents in a zone divided by the share of routine hours in it, and
    the same for the other accidents;
  * the mean chart probability over routine hours, the exposure-side
    average that Eq. (calib) would need for a per-flight-hour level.

Data: https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/ (cached under
data/isd_lite/) and isd-history.csv. Output: physics/out/metar_control.json
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import os
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.carb_icing_model import p_carb_icing, carb_ice_zone  # noqa: E402

KG = "event_extraction/out/causation_kg/factor_vectors.parquet"
DBS = ["data/NTSB_ASRS/avall.mdb", "data/NTSB_ASRS/Pre2008.mdb"]
CACHE = "data/isd_lite"
HISTORY = os.path.join(CACHE, "isd-history.csv")
OUT = "physics/out/metar_control.json"
ICE = "CARBURETOR_OR_INDUCTION_ICING"
N_OTHER = 2500
SEED = 0
N_BOOT_RR = 2000
MAX_KM = 60.0
HOUR_WINDOW = 3
BASE = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{y}/{u}-{w}-{y}.gz"


def f_to_c(f):
    return (f - 32.0) * 5.0 / 9.0


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def load_events():
    rows = []
    for db in DBS:
        p = subprocess.Popen(["mdb-export", db, "events"], stdout=subprocess.PIPE,
                             text=True, stderr=subprocess.DEVNULL)
        r = csv.reader(p.stdout)
        h = next(r)
        ix = {c: i for i, c in enumerate(h)}
        for row in r:
            g = lambda c: row[ix[c]].strip() if c in ix else ""
            try:
                t, d = float(g("wx_temp")), float(g("wx_dew_pt"))
            except ValueError:
                continue
            if (t == 0 and d == 0) or not (-60 < t < 130 and -60 < d < 130) or d > t + 2:
                continue
            try:
                dt = datetime.strptime(g("ev_date")[:8], "%m/%d/%y")
            except ValueError:
                continue
            hh = g("ev_time")
            hour = int(hh[:2]) if hh.isdigit() and len(hh) == 4 and int(hh[:2]) < 24 else None
            try:
                lat, lon = float(g("dec_latitude")), float(g("dec_longitude"))
            except ValueError:
                lat = lon = None
            rows.append({"ev_id": g("ev_id"), "year": dt.year, "month": dt.month,
                         "hour_utc": hour, "tz": g("ev_tmzn"), "fac": g("wx_obs_fac_id"),
                         "lat": lat, "lon": lon, "T": f_to_c(t), "Td": min(f_to_c(d), f_to_c(t))})
        p.wait()
    return pd.DataFrame(rows).drop_duplicates("ev_id")


def load_history():
    os.makedirs(CACHE, exist_ok=True)
    if not os.path.exists(HISTORY):
        urllib.request.urlretrieve("https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv", HISTORY)
    h = pd.read_csv(HISTORY, dtype=str)
    h = h[(h["CTRY"] == "US") & h["LAT"].notna() & h["LON"].notna()].copy()
    h["lat"] = h["LAT"].astype(float)
    h["lon"] = h["LON"].astype(float)
    h["begin"] = h["BEGIN"].astype(int) // 10000
    h["end"] = h["END"].astype(int) // 10000
    h["icao"] = h["ICAO"].fillna("").str.strip()
    h["wban_ok"] = h["WBAN"] != "99999"
    return h.reset_index(drop=True)


def candidates(ev, hist, k=3):
    """Nearest ISD stations covering the accident year, named facility first."""
    cov = hist[(hist["begin"] <= ev["year"]) & (hist["end"] >= ev["year"])]
    if ev["lat"] is None or np.isnan(ev["lat"]):
        return []
    d = haversine_km(ev["lat"], ev["lon"], cov["lat"].to_numpy(), cov["lon"].to_numpy())
    cov = cov.assign(km=d)
    cov = cov[cov["km"] <= MAX_KM].sort_values(["wban_ok", "km"], ascending=[False, True])
    if ev["fac"]:
        named = cov[cov["icao"] == "K" + ev["fac"].upper()]
        cov = pd.concat([named, cov.drop(named.index)])
    return [(r["USAF"], r["WBAN"], round(float(r["km"]), 1)) for _, r in cov.head(k).iterrows()]


def fetch(usaf, wban, year):
    path = os.path.join(CACHE, f"{usaf}-{wban}-{year}.gz")
    if os.path.exists(path):
        return path if os.path.getsize(path) > 0 else None
    try:
        urllib.request.urlretrieve(BASE.format(y=year, u=usaf, w=wban), path)
        return path
    except Exception:
        open(path, "wb").close()  # remember the miss
        return None


def read_lite(path):
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(io.StringIO(f.read()), sep=r"\s+", header=None, usecols=[0, 1, 2, 3, 4, 5],
                         names=["year", "month", "day", "hour", "t10", "d10"])
    df = df[(df["t10"] != -9999) & (df["d10"] != -9999)]
    df["T"] = df["t10"] / 10.0
    df["Td"] = np.minimum(df["d10"] / 10.0, df["T"])
    return df[["month", "day", "hour", "T", "Td"]]


def delong(y, s):
    pos, neg = y == 1, y == 0
    sp, sn = s[pos], s[neg]
    sn_s = np.sort(sn)
    v10 = (np.searchsorted(sn_s, sp, "left") + np.searchsorted(sn_s, sp, "right")) / (2 * len(sn))
    sp_s = np.sort(sp)
    v01 = 1 - (np.searchsorted(sp_s, sn, "left") + np.searchsorted(sp_s, sn, "right")) / (2 * len(sp))
    auc = float(v10.mean())
    se = float(np.sqrt(v10.var(ddof=1) / len(sp) + v01.var(ddof=1) / len(sn)))
    return {"auc": round(auc, 4), "ci95": [round(auc - 1.96 * se, 4), round(auc + 1.96 * se, 4)],
            "n_pos": int(pos.sum()), "n_neg": int(neg.sum())}


def zone_table(pi_acc, pi_hours):
    za = pd.Series([carb_ice_zone(p) for p in pi_acc]).value_counts(normalize=True)
    zh = pd.Series([carb_ice_zone(p) for p in pi_hours]).value_counts(normalize=True)
    return {z: {"share_accidents": round(float(za.get(z, 0.0)), 4),
                "share_hours": round(float(zh.get(z, 0.0)), 4),
                "relative_risk": round(float(za.get(z, 0.0) / zh.get(z, 1e-9)), 3)}
            for z in ("nil", "light", "moderate", "serious")}


def main():
    ev = load_events().set_index("ev_id")
    fv = pd.read_parquet(KG, columns=["record_id", ICE])
    fv["ev_id"] = fv["record_id"].str.rsplit("_", n=1).str[0]
    fv = fv.drop_duplicates("ev_id").set_index("ev_id")
    df = ev.join(fv[[ICE]], how="inner")
    df = df[df["lat"].notna()]
    icing = df[df[ICE] == 1]
    other = df[df[ICE] == 0].sample(n=min(N_OTHER, int((df[ICE] == 0).sum())), random_state=SEED)
    study = pd.concat([icing, other])
    print(f"study set: {len(icing)} icing accidents, {len(other)} other accidents", flush=True)

    hist = load_history()
    cand = {eid: candidates(r, hist) for eid, r in study.iterrows()}
    jobs = sorted({(u, w, int(study.loc[e, "year"])) for e, cs in cand.items() for (u, w, _) in cs[:1]})
    print(f"downloading {len(jobs)} station-years (first candidates) ...", flush=True)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: fetch(*j), jobs))

    rng = np.random.default_rng(SEED)
    rec = []
    cache = {}
    for eid, r in study.iterrows():
        hours = None
        for (u, w, km) in cand[eid]:
            key = (u, w, int(r["year"]))
            if key not in cache:
                p = fetch(*key)
                cache[key] = read_lite(p) if p else None
            lite = cache[key]
            if lite is not None:
                hours = lite[lite["month"] == int(r["month"])]
                if len(hours) >= 100:
                    station = (u, w, km)
                    break
                hours = None
        if hours is None:
            continue
        pi_all = np.array([p_carb_icing(t, d, "descent").p_ice for t, d in zip(hours["T"], hours["Td"])])
        if r["hour_utc"] is not None and not np.isnan(r["hour_utc"]):
            hh = int(r["hour_utc"])
            m = ((hours["hour"] - hh + 12) % 24 - 12).abs() <= HOUR_WINDOW
            pi_win = pi_all[m.to_numpy()]
        else:
            pi_win = pi_all
        rec.append({"ev_id": eid, "icing": int(r[ICE]), "station": station,
                    "pi_acc": p_carb_icing(r["T"], r["Td"], "descent").p_ice,
                    "pi_hours_all": pi_all, "pi_hours_window": pi_win,
                    "n_hours": int(len(pi_all)), "n_window": int(len(pi_win))})
    n_ice = sum(x["icing"] for x in rec)
    print(f"matched {len(rec)} accidents ({n_ice} icing) to routine hours", flush=True)

    out = {"n_icing_matched": n_ice, "n_other_matched": len(rec) - n_ice,
           "median_station_km": float(np.median([x["station"][2] for x in rec])),
           "hours_per_accident_median": float(np.median([x["n_hours"] for x in rec])),
           "window_hours_per_accident_median": float(np.median([x["n_window"] for x in rec]))}
    per_accident = {}
    for variant, key in (("all_hours", "pi_hours_all"), ("hour_matched", "pi_hours_window")):
        # equal weight per accident: 100 routine hours drawn per accident
        for label, flag in (("icing_vs_routine", 1), ("other_vs_routine", 0)):
            acc = np.array([x["pi_acc"] for x in rec if x["icing"] == flag])
            draws = [rng.choice(x[key], min(100, len(x[key])), replace=False)
                     if len(x[key]) > 0 else np.array([]) for x in rec if x["icing"] == flag]
            ctrl = np.concatenate([d for d in draws if len(d) > 0])
            if variant == "hour_matched":
                per_accident[flag] = (acc, draws)
            y = np.concatenate([np.ones(len(acc)), np.zeros(len(ctrl))])
            s = np.concatenate([acc, ctrl])
            out[f"{variant}/{label}"] = {**delong(y, s),
                                         "mean_pi_accidents": round(float(acc.mean()), 4),
                                         "mean_pi_routine": round(float(ctrl.mean()), 4),
                                         "zones": zone_table(acc, ctrl)}
    acc_i = np.array([x["pi_acc"] for x in rec if x["icing"] == 1])
    acc_o = np.array([x["pi_acc"] for x in rec if x["icing"] == 0])
    out["icing_vs_other_accidents_same_subset"] = delong(
        np.concatenate([np.ones(len(acc_i)), np.zeros(len(acc_o))]), np.concatenate([acc_i, acc_o]))
    # icing-specific relative risk by zone: RR(icing vs routine) / RR(other vs routine)
    zi = out["hour_matched/icing_vs_routine"]["zones"]
    zo = out["hour_matched/other_vs_routine"]["zones"]
    out["icing_specific_relative_risk_by_zone"] = {
        z: round(zi[z]["relative_risk"] / max(zo[z]["relative_risk"], 1e-9), 3) for z in zi}
    # bootstrap over accidents (each resampled with its own control hours),
    # icing and other accidents resampled independently (review 2026-09-30)
    zones = ("nil", "light", "moderate", "serious")
    zcode = {z: i for i, z in enumerate(zones)}

    def codes(v):
        return np.array([zcode[carb_ice_zone(p)] for p in v])

    enc = {}
    for flag in (1, 0):
        acc, draws = per_accident[flag]
        enc[flag] = (codes(acc), [codes(d) for d in draws])

    def rr(flag, idx):
        a, ds = enc[flag]
        sa = np.bincount(a[idx], minlength=4) / len(idx)
        h = np.concatenate([ds[i] for i in idx if len(ds[i]) > 0])
        sh = np.bincount(h, minlength=4) / len(h)
        return sa / np.maximum(sh, 1e-9)

    brng = np.random.default_rng(SEED + 1)
    n1, n0 = len(enc[1][0]), len(enc[0][0])
    boots = np.array([rr(1, brng.integers(0, n1, n1)) / np.maximum(rr(0, brng.integers(0, n0, n0)), 1e-9)
                      for _ in range(N_BOOT_RR)])
    out["icing_specific_relative_risk_by_zone_ci95"] = {
        z: [round(float(np.percentile(boots[:, i], 2.5)), 3),
            round(float(np.percentile(boots[:, i], 97.5)), 3)] for z, i in zcode.items()}
    out["icing_specific_relative_risk_n_boot"] = N_BOOT_RR
    os.makedirs("physics/out", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
