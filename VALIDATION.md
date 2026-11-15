# FlameGuard SI — validation

Generated **2026-11-15T11:33:49+00:00** · schema `2026-11-15.2` · data `c7646c87aac23ebe`
Engine **published-threshold margins** · model `published-threshold margins` · feature mode `margins_only` · endpoint `none`

> These metrics are measured on the held-out 30% only. Calibration is fitted on the
> 70% combiner-fit split; it is not inherited from the decision model. Published and derived labels
> are reported separately because FLEX/BASS numeric outcomes need a label policy.

## Gate and provenance

| item | value |
| --- | --- |
| labelled runs | **167** (gate 150) |
| reduced_claims | **False** |
| provenance_warning | **True** |
| published / derived | 24 / 143 |
| duplicate-state conflicts | 3 |

## Split

70/30 stratified by label; the six canonical cases are forced into test.

| split | n | class counts |
| --- | --- | --- |
| train | 116 | {'extinction': 27, 'marginal': 36, 'sustained_spread': 53} |
| test | 51 | {'extinction': 12, 'marginal': 16, 'sustained_spread': 23} |

## Held-out headline

| metric | value | 95% CI |
| --- | --- | --- |
| accuracy | **0.431** | [0.2941, 0.5686] |
| macro-F1 | **0.381** | [0.2277, 0.5211] |
| ECE (10 bins) | **0.2001** | — |
| Brier (macro) | **0.2159** | — |
| abstention rate | **0.392** | — |
| selective accuracy | **0.4516** | — |

| baseline | accuracy | macro-F1 |
| --- | --- | --- |
| majority (sustained_spread) | 0.451 | 0.207 |
| engine choice alone | n/a | n/a |

## Per class (held-out)

| class | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| extinction | 0.286 | 0.333 | 0.308 | 12 |
| marginal | 0.750 | 0.188 | 0.300 | 16 |
| sustained_spread | 0.455 | 0.652 | 0.536 | 23 |

## Confusion matrix (rows = actual, columns = predicted)

| actual \ predicted | extinction | marginal | sustained_spread |
| --- | --- | --- | --- |
| extinction | 4 | 0 | 8 |
| marginal | 3 | 3 | 10 |
| sustained_spread | 7 | 1 | 15 |

## Calibration (held-out)

- ECE: **0.2001** · Brier: **0.2159**

| bin | n | mean confidence | accuracy |
| --- | --- | --- | --- |
| 1 | 6 | 0.455 | 0.333 |
| 2 | 5 | 0.466 | 0.400 |
| 3 | 5 | 0.470 | 0.000 |
| 4 | 5 | 0.489 | 0.400 |
| 5 | 5 | 0.504 | 0.600 |
| 6 | 5 | 0.511 | 0.600 |
| 7 | 5 | 0.514 | 0.200 |
| 8 | 5 | 0.518 | 0.400 |
| 9 | 5 | 0.531 | 1.000 |
| 10 | 5 | 0.583 | 0.400 |

## Abstention (tau chosen on out-of-fold train only)

- tau = **0.40**, margin = **0.10**
- train review rate = 0.345, train selective accuracy = 0.592

| tau | coverage | review rate | accuracy on auto-clear |
| --- | --- | --- | --- |
| 0.40 | 0.655 | 0.345 | 0.592 **(chosen)** |
| 0.45 | 0.638 | 0.362 | 0.581 |
| 0.50 | 0.388 | 0.612 | 0.578 |
| 0.55 | 0.121 | 0.879 | 0.786 |
| 0.60 | 0.069 | 0.931 | 0.875 |
| 0.65 | 0.017 | 0.983 | 1.000 |
| 0.70 | 0.000 | 1.000 | 0.000 |
| 0.75 | 0.000 | 1.000 | 0.000 |

## Per family (held-out)

| family | n | accuracy | macro-F1 | abstention | note |
| --- | --- | --- | --- | --- | --- |
| BASS-II | 7 | 0.714 | 0.417 | 0.143 |  |
| BASS-SIBAL | 12 | 0.583 | 0.389 | 0.667 |  |
| FLEX | 25 | 0.280 | 0.272 | 0.440 |  |
| Saffire | 5 | 0.400 | 0.191 | 0.000 |  |
| SoFIE-GEL | 2 | 0.500 | 0.667 | 0.000 | insufficient n |

Macro-F1 in this table is computed over the classes present in the family, not a fixed 3-class macro.

## Published vs derived labels (held-out)

| basis | n | accuracy | macro-F1 | abstention |
| --- | --- | --- | --- | --- |
| derived | 42 | 0.429 | 0.335 | 0.405 |
| published | 9 | 0.444 | 0.416 | 0.333 |

## Misclassified examples

| run | family | actual | predicted | confidence | basis | source |
| --- | --- | --- | --- | --- | --- | --- |
| BASS-II-M19 | BASS-II | marginal | sustained_spread | 0.508 | derived | 20210011385 table 5.1 |
| BASS-II-M20 | BASS-II | marginal | sustained_spread | 0.471 | derived | 20210011385 table 5.1 |
| BASS-SIBAL-GMT100-T16 | BASS-SIBAL | sustained_spread | extinction | 0.446 | derived | 20210011385 table 7.1 |
| BASS-SIBAL-GMT178-T11 | BASS-SIBAL | sustained_spread | extinction | 0.468 | derived | 20210011385 table 7.1 |
| BASS-SIBAL-GMT190-T24 | BASS-SIBAL | marginal | sustained_spread | 0.447 | derived | 20210011385 table 7.1 |
| BASS-SIBAL-GMT216-T28a | BASS-SIBAL | extinction | sustained_spread | 0.514 | published | 20210011385 table 7.1 |
| BASS-SIBAL-GMT45-T4 | BASS-SIBAL | extinction | sustained_spread | 0.460 | published | 20210011385 table 7.1 |
| FLEX-001 | FLEX | marginal | sustained_spread | 0.520 | derived | 20150023456 table IX |
| FLEX-020 | FLEX | sustained_spread | marginal | 0.677 | derived | 20150023456 table IX |
| FLEX-062 | FLEX | marginal | sustained_spread | 0.517 | derived | 20150023456 table IX |

## Limitations

- The 167 rows are a convenience extraction, not a random sample of all microgravity combustion data.
- 143 of 167 labels are derived from published numeric outcomes (FLEX burn time, BASS-II spread rate, Saffire burn length); only 24 are explicitly categorical. The published/derived table above is the honest split.
- The engine contract includes a microgravity reference with the published thresholds, and the four margin features are computed from the same policy. Derived-label metrics therefore measure policy agreement, not independent prediction; the 24 published labels are the independent check.
- FLEX `Extinction` is the normal end of a droplet burn; it is not a failed ignition. The burn-time rule is documented in `data/LABEL_POLICY.md`.
- SoFIE GEL rows are extinction-point conditions, not initial conditions; they are limit tests.
- The held-out set is small (about 50 rows), so the confidence intervals are wide.
- The decision model is frozen; its own confidence is not calibrated for this task. The combiner is calibrated on the 70% split.
- Suppression strategies are a deterministic policy from published fire-safety guidance, not a learned classifier, and are not validated here.
- Nothing here is a flight-qualified procedure.

## Reproduce

```bash
python -m flameguard.evaluate --no-engine
```
