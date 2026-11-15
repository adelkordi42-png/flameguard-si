# Sources and extraction

Every row in `data/runs.jsonl` is traceable to a public NASA NTRS document and a
table. `outcome_quote` preserves the source line so a reviewer can audit the
label without trusting the extraction script.

| family | NTRS ID | table | rows | URL |
| --- | --- | --- | --- | --- |
| BASS-II | 20210011385 | 5.1 | 20 | https://ntrs.nasa.gov/api/citations/20210011385 |
| BASS SIBAL | 20210011385 | 7.1 | 31 | https://ntrs.nasa.gov/api/citations/20210011385 |
| Saffire I/II | 20170008805 | I | 11 | https://ntrs.nasa.gov/api/citations/20170008805 |
| SoFIE GEL | 20240007628 | 1 | 5 | https://ntrs.nasa.gov/api/citations/20240007628 |
| FLEX | 20150023456 | IX | 100 | https://ntrs.nasa.gov/api/citations/20150023456 |

## Extraction method

- **BASS-II 5.1:** typed from the published test table. Material PMMA, pressure
  101.3 kPa, Kanthal hot-wire igniter, microgravity. `thickness_mm` and initial
  O2 are the table columns; `opposed_flow_cm_s` is the first value in the flow
  sequence. Label is derived from the spread rate by the locked policy.
- **BASS SIBAL 7.1:** typed from the published test summary. SIBAL is a 75%
  cotton / 25% fiberglass fabric, 0.32 mm thick; pressure 101.3 kPa; hot-wire
  igniter. Comments map to labels as documented.
- **Saffire I/II Table I:** 11 samples. Saffire 1 is 21.7% O2, Saffire 2 is
  22.1% O2, pressure 101.3 kPa, flow 20 cm/s (25 cm/s for sample 2-6). Burn
  duration and length are the published outcome; labels follow the policy.
- **SoFIE GEL Table 1:** five extinction-limit cases on a 40 mm PMMA sphere.
  O2, flow, and pressure are the extinction-point values printed in the table;
  the report states this caveat.
- **FLEX Table IX:** parsed from the 41 MB NTRS PDF with `pdfplumber` because the
  plain-text dump is column-major. 276 tests were parsed; 8 lacked a burn time or
  pressure and were excluded. A deterministic, label-stratified sample of 100
  was selected across the O2/pressure/fuel envelope. The FLEX source line is
  preserved as `outcome_quote`. The PDF is not committed; the script downloads
  it on demand.

## Reproduce

```bash
python scripts/build_runs.py --flex-pdf /tmp/flex.pdf
python -m flameguard.dataset --audit
```

The PDF parser is a build-time tool only. It is not imported by the service and
`pdfplumber` is not in `requirements.txt`.

## TypeSafe contract

The brief cited `https://beta.typesafe.ai/docs/jev/get-started` and
`POST /v1/jev:run`. On 2026-11-15 the docs host does not resolve and the endpoint
returns 404. The live contract is:

- `GET https://api.typesafe.ai/openapi.json` (TypeSafe 0.2.0)
- `POST https://api.typesafe.ai/v1/systemone`
- `GET https://api.typesafe.ai/v1/models`
- `Authorization: Bearer $JEV_API_KEY`

The OpenAPI document is committed at `docs/typesafe-openapi.json` for audit.