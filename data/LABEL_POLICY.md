# Label policy — locked before fitting

This policy was written before any combiner fit and is applied by
`scripts/build_runs.py`. Every row in `data/runs.jsonl` carries `label`,
`label_basis`, `outcome_raw`, and `outcome_quote`. A row is never relabelled to
improve a metric.

## Classes

| label | meaning |
| --- | --- |
| `sustained_spread` | Established ignition and self-sustaining burning or spread. |
| `marginal` | Ignites, but the behaviour is near-limit, partial, intermittent, one-sided, or disruptive; persistence is genuinely ambiguous. |
| `extinction` | No ignition, quench, blow-off, or flame loss before established burning. |

## Tier A — source states a categorical outcome

- **BASS SIBAL (Table 7.1):** `Quenched` / `No ignition` / `did not ignite` /
  `Blow-off` → `extinction`; `No blowoff` → `sustained_spread`. These are
  `label_basis = published`.
- **Saffire I/II (Table I):** `insignificant` burn, `~0` burn length → `extinction`
  (`published`). Sustained and partial outcomes are derived from the reported
  burn duration/length/spread and marked `derived`.
- **SoFIE GEL (Table 1):** the five cases are extinction-limit tests
  (`quenching`, `blowoff`, `intermediate`); all are `extinction`, `published`.
  The conditions are extinction-point values, not initial conditions — a caveat
  reported in `VALIDATION.md`.

## Tier B — label derived from a reported number

- **BASS-II (Table 5.1):** PMMA tests report spread rate and burn time. A test is
  `marginal` when its reported spread rate is `<0.02 mm/s` (near-limit), else
  `sustained_spread`; `Not tracked` with a positive burn time is
  `sustained_spread`. All are `derived`.
- **FLEX (Table IX):** droplets are not a 3-class outcome. A test is
  `extinction` when burn time `<2 s`, `marginal` when the test end is
  `Disruption` or burn time is `2–5 s`, and `sustained_spread` when the test end
  is `Extinction`/`Completion` with burn time `>=5 s`. FLEX `Extinction` is the
  normal end of a droplet burn; it is **not** mapped to the target `extinction`
  without the burn-time rule. All are `derived`.
- **Saffire / BASS-II outcomes without an explicit categorical comment** are
  derived from burn duration, burn length, or spread rate and marked `derived`.

## Gate and provenance

- The hard gate is 150 complete rows. Below 150, `reduced_claims=true`; the
  report is still generated.
- `provenance_warning=true` when fewer than 50 labels are published or more than
  half are derived. This is a disclosure, not a failure of the 150 gate.
- No synthetic row, no imputed condition, and no row without a source quote is
  admitted. Duplicate condition states with conflicting labels are counted by
  `python -m flameguard.dataset --audit` and reported, not silently merged.

## Why FLEX is included

FLEX supplies the low-O2, high-diluent, low-pressure corner that BASS/Saffire do
not cover. Its labels are proxies, so the validation report prints published and
derived metrics separately and never calls a derived label a “NASA outcome”.