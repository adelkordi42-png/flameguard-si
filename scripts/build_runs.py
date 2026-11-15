#!/usr/bin/env python3
"""Build data/runs.jsonl from the NTRS tables cited in data/sources.md.

This is a build-time tool, not part of the runtime service. It exists so every
row can be regenerated and re-audited. FLEX is parsed from the published PDF
(Table IX); the other four tables are small enough to encode verbatim with the
source quote that justifies each label.

    python scripts/build_runs.py --flex-pdf /tmp/flex.pdf

Requires the dev-only `pdfplumber` package for FLEX.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "runs.jsonl"
FLEX_URL = "https://ntrs.nasa.gov/api/citations/20150023456/downloads/20150023456.pdf"


def row(**kw):
    base = {
        "material": None,
        "geometry": "flat_sheet",
        "thickness_mm": None,
        "o2_percent": None,
        "pressure_kpa": 101.3,
        "opposed_flow_cm_s": None,
        "igniter": "hot_wire",
        "gravity": "microgravity",
        "notes": "",
    }
    base.update(kw)
    assert base["label"] in {"sustained_spread", "marginal", "extinction"}, base
    assert base["label_basis"] in {"published", "derived"}, base
    return base


# ---------------------------------------------------------------- BASS-II 5.1
# Test, thickness mm, width cm, sides, flow sequence cm/s, spread mm/s, burn min,
# initial O2 %, final O2 %.  Table 5.1, NASA/TM-20210011385 p. 58.
BASS = [
    ("M1", 1.0, 2.2, 1, "9", 0.071, 6.4, 22.2, 21.9),
    ("M2", 3.0, 2.2, 1, "20 and 15", 0.052, 22.9, 22.2, 21.2),
    ("M3", 5.0, 2.2, 1, "10 and 20", 0.029, 20.1, 20.8, 19.7),
    ("M4", 5.0, 2.2, 2, "5 and 3", 0.045, 19.8, 21.8, 19.8),
    ("M5", 3.0, 2.2, 2, "10", 0.071, 27.2, 22.2, 19.0),
    ("M6", 1.0, 2.2, 2, "10 and 8", None, 16.8, 21.8, 20.7),
    ("M7", 1.0, 1.2, 2, "10, 6, 5, and 3", 0.070, 18.4, 18.9, 18.4),
    ("M8", 2.0, 1.2, 2, "10, 6, and 5", 0.040, 25.2, 19.1, 18.2),
    ("M9", 3.0, 1.2, 2, "14", 0.025, 16.8, 18.0, 17.6),
    ("M10", 4.0, 1.2, 2, "15 and 10", 0.030, 30.5, 20.1, 18.6),
    ("M11", 5.0, 1.2, 2, "21, 15, and 13", 0.022, 28.5, 20.0, 18.5),
    ("M12", 2.0, 2.2, 2, "12 and 10", None, 16.2, 21.7, 20.4),
    ("M13", 4.0, 2.2, 2, "6, 5, and 4", 0.033, 20.5, 20.8, 19.0),
    ("M14", 2.0, 2.2, 1, "15, 14, and 10", 0.0353, 15.4, 22.2, 21.6),
    ("M15", 4.0, 2.2, 1, "20", 0.045, 22.0, 21.0, 20.0),
    ("M16", 1.0, 2.2, 2, "5, 3, and 2", 0.1062, 8.4, 19.2, 18.7),
    ("M17", 2.0, 2.2, 2, "5, 4, and 3", 0.026, 30.7, 18.8, 17.1),
    ("M18", 3.0, 2.2, 2, "10 and 8", 0.013, 25.5, 17.9, 16.8),
    ("M19", 4.0, 2.2, 2, "10 and 6", 0.0159, 25.7, 19.0, 17.4),
    ("M20", 5.0, 2.2, 2, "10", 0.012, 16.5, 18.0, 17.4),
]


def bass_rows():
    out = []
    for name, thick, width, sides, flow_seq, spread, burn, o2, o2_final in BASS:
        first_flow = float(re.findall(r"\d+(?:\.\d+)?", flow_seq)[0])
        label = "marginal" if spread is not None and spread < 0.02 else "sustained_spread"
        spread_text = "not tracked" if spread is None else f"{spread:g} mm/s"
        out.append(
            row(
                run_id=f"BASS-II-{name}",
                family="BASS-II",
                source={
                    "doc_id": "20210011385",
                    "table": "5.1",
                    "url": "https://ntrs.nasa.gov/api/citations/20210011385",
                    "page": 58,
                },
                material="PMMA",
                geometry="flat_sheet",
                thickness_mm=thick,
                o2_percent=o2,
                pressure_kpa=101.3,
                opposed_flow_cm_s=first_flow,
                outcome_raw=f"spread_rate={spread_text}; burn_time={burn} min; O2_final={o2_final}%",
                outcome_quote=(
                    f"{name} {thick:g} {width} {sides} {flow_seq} {spread_text} {burn} {o2} {o2_final}"
                ),
                label=label,
                label_basis="derived",
                notes=f"width={width} cm; sides={sides}; flow sequence={flow_seq}; O2 initial {o2}%",
            )
        )
    return out


# ------------------------------------------------------------- BASS SIBAL 7.1
# Test, width cm, flow cm/s, O2 %, comment.  Table 7.1, p. 101.
SIBAL = [
    ("GMT45-T1", 2.2, "10", 18.5, ""),
    ("GMT45-T2", 2.2, "10 to 5", 18.5, ""),
    ("GMT45-T4", 2.2, "10 to 2.2", 18.7, "Quenched"),
    ("GMT45-T15", 2.2, "10 to 29", 18.7, "No blowoff"),
    ("GMT100-T5", 2.2, "10 to 2.4", 18.8, ""),
    ("GMT100-T6", 2.2, "4.5 to 2.6", 18.8, ""),
    ("GMT100-T13", 2.2, "4 to 2.2", 17.5, "Quenched"),
    ("GMT100-T16", 2.2, "4 to 3", 17.6, ""),
    ("GMT175-T9", 2.2, "10", 16.4, "No ignition"),
    ("GMT175-T10", 2.2, "5", 16.4, "No ignition"),
    ("GMT175-T18", 2.2, "5 to 2.6", 17.4, "Quenched"),
    ("GMT178-T11", 2.2, "4", 17.1, ""),
    ("GMT178-T12", 2.2, "5", 16.8, "No ignition"),
    ("GMT178-T14", 2.2, "4 to 2.8", 16.9, "Quenched"),
    ("GMT178-T17", 2.2, "6 to 53", 16.9, "No blowoff"),
    ("GMT190-T19", 1.2, "11", 17.2, ""),
    ("GMT190-T20", 1.2, "11 to 3", 17.2, "Quenched"),
    ("GMT190-T21", 1.2, "11 to 47", 17.2, "Blow-off"),
    ("GMT190-T22", 1.2, "10.5 to 5", 17.1, "Quenched"),
    ("GMT190-T23", 2.2, "10", 17.2, "Reused, did not ignite"),
    ("GMT190-T24", 2.2, "10", 18.2, "Reused, one-sided flame"),
    ("GMT216-T25", 2.2, "5 to 2.5", 20.8, "Reused, ignited"),
    ("GMT216-T26", 2.2, "5 to 2.5", 20.8, "Reused, ignited"),
    ("GMT216-T27", 1.2, "5 to 2", 20.7, "Reused, ignited"),
    ("GMT216-T28a", 1.2, "5", 20.7, "Reused, did not ignite"),
    ("GMT216-T28b", 1.2, "5", 20.7, "Reused, did not ignite"),
    ("GMT216-T29", 1.2, "5 to 2", 20.7, "Reused, ignited"),
    ("GMT96-T8", 2.2, "5", 21.0, "BASS"),
    ("GMT96-T7", 2.2, "10", 21.0, "BASS"),
    ("GMT131-T10", 1.2, "11", 21.0, "BASS"),
    ("GMT222-T11", 1.2, "19", 21.0, "BASS"),
]


def sibal_rows():
    out = []
    for name, width, flow_seq, o2, comment in SIBAL:
        first_flow = float(re.findall(r"\d+(?:\.\d+)?", flow_seq)[0])
        lowered = comment.lower()
        if any(k in lowered for k in ("quench", "no ignition", "did not ignite", "blow-off")):
            label, basis = "extinction", "published"
        elif "one-sided" in lowered:
            label, basis = "marginal", "derived"
        elif "no blowoff" in lowered:
            label, basis = "sustained_spread", "published"
        else:
            label, basis = "sustained_spread", "derived"
        out.append(
            row(
                run_id=f"BASS-SIBAL-{name}",
                family="BASS-SIBAL",
                source={
                    "doc_id": "20210011385",
                    "table": "7.1",
                    "url": "https://ntrs.nasa.gov/api/citations/20210011385",
                    "page": 101,
                },
                material="SIBAL_fabric",
                geometry="flat_sheet",
                thickness_mm=0.32,
                o2_percent=o2,
                pressure_kpa=101.3,
                opposed_flow_cm_s=first_flow,
                outcome_raw=comment or "blank comment (normal burn)",
                outcome_quote=f"{name} {width} {flow_seq} {o2} {comment}".strip(),
                label=label,
                label_basis=basis,
                notes=f"width={width} cm; flow sequence={flow_seq}; SIBAL 75% cotton / 25% fiberglass, 0.32 mm",
            )
        )
    return out


# --------------------------------------------------------------- Saffire I/II
# Table I, NASA NTRS 20170008805 p. 28.  Values are the published sample table.
SAFFIRE = [
    ("1-1", "SIBAL_fabric", 40.6, 0.37, 420, "~84 cm", 20, "concurrent", 21.7, "1.8 mm/s", "sustained_spread", "derived"),
    ("1-2", "SIBAL_fabric", 40.6, 0.37, 70, "~10 cm", 20, "opposed", 21.7, "1.3 mm/s", "sustained_spread", "derived"),
    ("2-1", "silicone", 5.0, 0.27, 0, "~0", 20, "concurrent", 22.1, "n/a", "extinction", "published"),
    ("2-2", "silicone", 5.0, 0.61, 0, "~0", 20, "concurrent", 22.1, "n/a", "extinction", "published"),
    ("2-3", "silicone", 5.0, 1.03, 0, "~0", 20, "concurrent", 22.1, "n/a", "extinction", "published"),
    ("2-4", "silicone", 5.0, 0.37, 0, "~0", 20, "opposed", 22.1, "n/a", "extinction", "published"),
    ("2-5", "SIBAL_fabric", 5.0, 0.37, 145, "29 cm", 20, "concurrent", 22.1, "2.1 mm/s", "sustained_spread", "derived"),
    ("2-6", "SIBAL_fabric", 5.0, 0.37, 115, "29 cm", 25, "concurrent", 22.1, "2.6 mm/s", "sustained_spread", "derived"),
    ("2-7", "pmma_nomex", 5.0, 0.85, 140, "5 cm PMMA; Nomex not ignited", 20, "concurrent", 22.1, "n/a", "marginal", "derived"),
    ("2-8", "PMMA", 5.0, 10.0, 600, "~10 cm", 20, "concurrent", 22.1, "note iv", "marginal", "derived"),
    ("2-9", "PMMA", 5.0, 10.0, 900, "~10 cm", 20, "concurrent", 22.1, "note iv", "marginal", "derived"),
]


def saffire_rows():
    out = []
    for sample, material, width, thick, duration, length, flow, direction, o2, spread, label, basis in SAFFIRE:
        out.append(
            row(
                run_id=f"Saffire-{sample}",
                family="Saffire",
                source={
                    "doc_id": "20170008805",
                    "table": "I",
                    "url": "https://ntrs.nasa.gov/api/citations/20170008805",
                    "page": 28,
                },
                material=material,
                geometry="flat_sheet",
                thickness_mm=thick,
                o2_percent=o2,
                pressure_kpa=101.3,
                opposed_flow_cm_s=flow,
                outcome_raw=f"burn_duration={duration} s; burn_length={length}; spread_rate={spread}; direction={direction}",
                outcome_quote=(
                    f"{sample} {material} {width} cm {thick} mm; {duration} s; {length}; {flow} cm/s; O2 {o2}%"
                ),
                label=label,
                label_basis=basis,
                notes=f"width={width} cm; direction={direction}; Saffire I/II cabin air",
            )
        )
    return out


# ---------------------------------------------------------------- SoFIE GEL 1
# Table 1, NASA NTRS 20240007628 p. 35.  Extinction-point conditions, not initial.
SOFIE = [
    (1, "1409B", "quenching", "inside duct", 4.83, 128.8, 3.63, 14.21, 1.04, 0.36),
    (2, "1408A", "blowoff", "inside duct", 34.28, 914.1, 25.71, 14.99, 1.06, 0.95),
    (3, "1402C", "intermediate", "inside duct", 11.50, 306.7, 8.63, 13.54, 1.07, 0.37),
    (4, "1409C", "quenching", "duct exit", 0.44, 11.7, 0.33, 20.77, 1.03, 0.17),
    (5, "1410C", "blowoff", "duct exit", 49.68, 1324.8, 37.26, 14.37, 1.07, None),
]


def sofie_rows():
    out = []
    for case, sample, mode, location, flow, reynolds, strain, o2, pressure_atm, heat_loss in SOFIE:
        out.append(
            row(
                run_id=f"SoFIE-GEL-{case}",
                family="SoFIE-GEL",
                source={
                    "doc_id": "20240007628",
                    "table": "1",
                    "url": "https://ntrs.nasa.gov/api/citations/20240007628",
                    "page": 35,
                },
                material="PMMA",
                geometry="sphere",
                thickness_mm=40.0,
                o2_percent=o2,
                pressure_kpa=round(pressure_atm * 101.325, 2),
                opposed_flow_cm_s=flow,
                outcome_raw=(
                    f"extinction_mode={mode}; location={location}; flow={flow} cm/s; "
                    f"Re={reynolds}; strain={strain} 1/s; O2={o2}%; heat_loss={heat_loss}"
                ),
                outcome_quote=(
                    f"{case} {sample} {mode} {location} {flow} {reynolds} {strain} {o2} "
                    f"{pressure_atm} {heat_loss if heat_loss is not None else 'not recorded'}"
                ),
                label="extinction",
                label_basis="published",
                notes="extinction-point conditions from Table 1, not initial conditions; 40 mm PMMA sphere",
            )
        )
    return out


# --------------------------------------------------------------------- FLEX IX
def flex_rows(pdf_path: Path, target: int = 100):
    import pdfplumber

    parsed = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                if not re.match(r"^FLEX[–-]\d{3}\s", line):
                    continue
                parts = line.replace("–", "-").split()
                if len(parts) != 15:
                    continue
                (rid, ident, date, gmt, fuel, pressure, o2, n2, co2, he,
                 d0, dext, rate, burn, end) = parts
                try:
                    pressure_mmhg = float(pressure)
                    o2_frac = float(o2)
                    d0_mm = float(d0)
                    burn_s = float(burn)
                except ValueError:
                    continue
                if pressure_mmhg <= 0 or d0_mm <= 0:
                    continue
                if end == "Disruption":
                    label = "marginal"
                elif burn_s < 2.0:
                    label = "extinction"
                elif burn_s < 5.0:
                    label = "marginal"
                else:
                    label = "sustained_spread"
                parsed.append(
                    row(
                        run_id=f"FLEX-{rid.split('-')[1]}",
                        family="FLEX",
                        source={
                            "doc_id": "20150023456",
                            "table": "IX",
                            "url": "https://ntrs.nasa.gov/api/citations/20150023456",
                            "page": None,
                        },
                        material="methanol" if fuel.lower() == "methanol" else "heptane",
                        geometry="droplet",
                        thickness_mm=d0_mm,
                        o2_percent=round(o2_frac * 100.0, 2),
                        pressure_kpa=round(pressure_mmhg * 0.133322, 2),
                        opposed_flow_cm_s=0.0,
                        outcome_raw=(
                            f"test_end={end}; burn_time_s={burn_s}; D0_mm={d0_mm}; "
                            f"D_ext_mm={dext}; burning_rate_mm2_s={rate}"
                        ),
                        outcome_quote=line,
                        label=label,
                        label_basis="derived",
                        notes=(
                            "quiescent droplet; thickness_mm carries the initial droplet diameter; "
                            "FLEX Extinction is normal burn-out unless burn time <2 s"
                        ),
                    )
                )
    if not parsed:
        raise SystemExit("no FLEX rows parsed; check the PDF and pdfplumber")
    buckets = defaultdict(list)
    for item in parsed:
        buckets[item["label"]].append(item)

    def pick(items, k):
        items = sorted(
            items,
            key=lambda r: (r["o2_percent"], r["pressure_kpa"], r["material"], r["run_id"]),
        )
        if len(items) <= k:
            return items
        if k == 1:
            return [items[len(items) // 2]]
        idx = sorted({round(i * (len(items) - 1) / (k - 1)) for i in range(k)})
        return [items[i] for i in idx]

    extinction = buckets["extinction"]
    marginal = pick(buckets["marginal"], min(45, len(buckets["marginal"])))
    sustained_n = max(0, target - len(extinction) - len(marginal))
    sustained = pick(buckets["sustained_spread"], sustained_n)
    selected = extinction + marginal + sustained
    if len(selected) < target:
        # Top up from the largest remaining bucket without changing the policy.
        used = {r["run_id"] for r in selected}
        spare = [r for r in parsed if r["run_id"] not in used]
        selected += pick(spare, target - len(selected))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flex-pdf", type=Path, default=Path("/tmp/flex.pdf"))
    parser.add_argument("--flex-count", type=int, default=100)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if not args.flex_pdf.exists():
        print(f"downloading {FLEX_URL}")
        urllib.request.urlretrieve(FLEX_URL, args.flex_pdf)

    rows = bass_rows() + sibal_rows() + saffire_rows() + sofie_rows() + flex_rows(args.flex_pdf, args.flex_count)
    ids = [r["run_id"] for r in rows]
    if len(ids) != len(set(ids)):
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        raise SystemExit(f"duplicate run_id: {dupes}")
    for r in rows:
        if r["o2_percent"] is None or r["pressure_kpa"] is None or r["opposed_flow_cm_s"] is None:
            raise SystemExit(f"incomplete conditions: {r['run_id']}")
        if r["thickness_mm"] is None or r["thickness_mm"] <= 0:
            raise SystemExit(f"missing thickness/diameter: {r['run_id']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in sorted(rows, key=lambda x: (x["family"], x["run_id"])):
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    labels = Counter(r["label"] for r in rows)
    basis = Counter(r["label_basis"] for r in rows)
    families = Counter(r["family"] for r in rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    print("labels:", dict(labels))
    print("basis:", dict(basis))
    print("families:", dict(families))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())