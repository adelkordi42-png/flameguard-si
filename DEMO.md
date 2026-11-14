# Judge demo script — 90 seconds

**URL:** <https://flameguard-si.onrender.com>

Before you walk over, warm the service — Render sleeps the free tier after 15
minutes idle and the first request then takes ~1 minute:

```bash
curl -s https://flameguard-si.onrender.com/api/health
```

Stop when it prints `{"status":"ok",...}`.

---

## Beat 1 — the problem (15 s)

> "Spacecraft fire safety is a life-or-death call, and NASA has decades of
> microgravity combustion data — BASS, FLEX, Saffire — sitting in a database.
> The problem is that data tells you what *happened*; it doesn't tell you what to
> *do* about the cabin in front of you."

Page loads with a verdict already rendered, so there is no empty screen.

## Beat 2 — a normal call (20 s)

Press **"Oxygen-rich, quiescent flow"**.

Expect: **Sustained spread**, calibrated confidence **85.6%**, and a suppression
strategy recommending inerting plus depressurisation.

> "One click, under a second, and it doesn't just label the flame — it names the
> suppression action and the O2 and pressure regime it applies to."

Point at **Section 3**: the five nearest labelled runs, each with the outcome
actually recorded, and an "agrees with verdict" column.

## Beat 3 — the opposite regime (15 s)

Press **"Inerted, high opposed flow"**.

Expect: **Extinction**, **85.2%**, "no active suppression required — maintain
inerting margin and monitor".

> "Same model, opposite regime. The opposed flow blows the flame out before it
> can establish."

## Beat 4 — the differentiator (25 s)

Set **Material: `cotton`**, **O2: `15.78`**, **Pressure: `75.25`**,
**Opposed flow: `0`**, **Geometry: `confined`**. Assess.

Expect: **Needs human review**, confidence **50.8%**, reason `low_confidence`,
and the suppression line replaced by an escalation instruction.

> "This is the part that makes it usable on a spacecraft. At 51% confidence the
> system refuses to give a verdict. It says *I don't know* and hands the decision
> to a human. A dashboard that always answers is a dashboard you stop trusting."

Then mention the other two abstention triggers: top-two classes too close, and
any input outside the trained range (`o2 = 5000` → `input_outside_observed_range`).

## Beat 5 — honesty (15 s)

> "The model is calibrated, and it's validated on held-out runs with a reported
> confusion matrix and Brier score. Right now it's trained on a synthetic
> placeholder while the real PSI measurement files are being pulled in — the
> badge at the top says so. Swapping in the real BASS runs is one command; the
> eval harness and the UI don't change."

---

## If something breaks

| Symptom | Cause | Recovery |
| --- | --- | --- |
| First request hangs ~60 s | free-tier cold start | it recovers by itself; that's why the warm-up curl exists |
| Page shows "model unavailable" | deploy restarted mid-demo | wait 60 s, reload |
| A preset gives a different class | model retrained on new data | read the verdict card aloud; the class is what the model says, not what this script says |

## Numbers to have ready

- Held-out accuracy **0.794**, macro-F1 **0.788** vs majority baseline **0.185**
- Brier **0.1133 → 0.1059** after Platt calibration
- Abstention τ **0.55**, review rate **21.7%**, accuracy on auto-clear **85.8%**
- Production p95 latency **314 ms** (≈260 ms of that is network round-trip)