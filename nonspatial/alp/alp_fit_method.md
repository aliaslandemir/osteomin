# ALP Fit Method

## Model
\[
\frac{dP}{dt} = r_b^P + u_{\mathrm{osteo}}(t)\,r_o^P
\]

Control condition uses only basal production:
\[
P_{\mathrm{CON}}(t) = P_0 + r_b^P\,t_h
\]

Osteogenic condition uses basal + osteogenic production:
\[
P_{\mathrm{OS}}(t) = P_0 + \left(r_b^P + r_o^P\right)t_h
\]

## Parameters
- Fitted: `P0`, `r_basal_P`, `r_osteo_P`
- Units: assay-units/hour for rates

## Data and Horizon
- Timepoints: day 3, 7, 14, 21
- Default output horizon: day 21

## Optimization
- Primary: `scipy.optimize.least_squares`
- Fallback: linear least squares (`numpy.linalg.lstsq`)
- Objective: SSE across CON and OS observations
