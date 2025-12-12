# AA → Collagen (Michaelis–Menten, 24 h exposure) — minimal report

## Inputs
- AA0: 50.0 µg/mL
- Km: 5.0 µg/mL
- AA half-life: 3.0 h
- Medium interval: 2 days
- Supplements start day: 3
- Simulated days: 14
- Measurement day: 14
- Active-window mean shown: first 3 h per day
- BASAL endpoint at day 14: 1.36
- AA (osteo) endpoint at day 14: 1.58

## Fitted parameters (from endpoints, fixed Km)
- R0: 0.0971429 (basal collagen/day)
- R_max: 0.0543634 (max AA-driven collagen/day)

## Diagnostics
- max AA_daily_24h: 10.0591 µg/mL
- max AA_daily_3h: 40.3944 µg/mL
- sat_24h range: [0, 0.667976]
- sum sat_24h[1..D]: 4.04684
- max AA-driven rate: 0.0363134

## Endpoint consistency
- Model basal at day 14: 1.36
- Model osteo at day 14: 1.58
- Exp ratio (AA/BASAL): 1.16176
- Model ratio (AA/BASAL): 1.16176

## Files
- `aa_collagen_output.xlsx`
- `basal_vs_aa_better.pdf` / `basal_vs_aa_better.png`
- `aa_exposure_daily.pdf` / `aa_exposure_daily.png`
- `mm_curve.pdf` / `mm_curve.png`
