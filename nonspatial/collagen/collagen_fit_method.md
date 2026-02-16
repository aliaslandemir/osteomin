# Collagen Fit Method

## Model
\[
\frac{dC}{dt} = r_b^C + u_{\mathrm{osteo}}(t)\,r_o^C\,\frac{A(t)}{K_A^C + A(t)}
\]

AA pulse-decay protocol:
\[
A(t)=
\begin{cases}
0, & t<t_{\mathrm{osteo}}\\
A_0\exp\left[-k_A(t-t_n)\right], & t_n\le t < t_{n+1}
\end{cases}
\quad
k_A=\ln(2)/\tau_A
\]
with refresh times:
\[
t_n=t_{\mathrm{osteo}}+n\,\Delta t_{\mathrm{med}}
\]

## Parameters
- Fitted: `C0`, `r_basal_C`, `r_osteo_C`
- Fixed per run: `K_A`, `A0`, `tau_A_h`, `dt_med_h`, `t_osteo_h`
- Units: assay-units/hour for rates

## Data and Horizon
- Timepoints: day 1, 2, 4, 6, 8, 10, 12
- Default output horizon: day 12

## Optimization
- Primary: `scipy.optimize.least_squares`
- Fallback: bounded grid search
- Objective: SSE across control and AA observations
