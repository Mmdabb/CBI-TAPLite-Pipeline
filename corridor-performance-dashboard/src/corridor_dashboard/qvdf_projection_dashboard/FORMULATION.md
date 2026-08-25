# Integrated CBI and TAPLite QVDF reconstruction

This document defines the two curves shown in each integrated dashboard
projection panel:

1. the CBI reconstruction from accepted sensor episodes; and
2. the TAPLite assignment projection from period link-performance results.

For ordered congestion boundaries, both curves use the same queue-to-speed
conversion and asymmetric QVDF queue shape inside an episode, with smoothstep
anchors outside `t0` and `t3`. When TAPLite emits unordered or collapsed
boundaries, the assignment curve uses TAPLite's reprojected `spd_mph_*`
series directly.

## 1. Inputs and precedence

For each corridor TMC, mapped network link, and AM/MD/PM period, the dashboard
loads:

| value | source |
|---|---|
| observed profile and free-flow context | CBI average-weekday profile |
| accepted `t0`, `t2`, `t3`, minimum speed, cutoff, length, discharge | CBI accepted episode |
| calibrated `f_d`, `n`, `f_p`, `s` | CBI selected QVDF parameters |
| volume, capacity, `P`, `t0`, `t2`, `t3`, `vt2`, free speed, cutoff, discharge | TAPLite `link_performance.csv` |
| assignment time-dependent speed | TAPLite `link_performance.csv` `spd_mph_*` columns |

The canonical TMC-to-link artifact produced by the CBI run is used throughout
the corridor and dashboard pipeline. For multiple candidates, mapped distance
is preferred when it is available; otherwise source occurrence order is used.

One row is retained for every observed TMC and period. The TMC's canonical
primary mapped link is used when it has a usable assignment curve; otherwise
the next ranked mapped link with a usable curve is selected. A shared network
link may supply assignment attributes to several observed TMCs.

TAPLite `link_performance.csv` is the only assignment-curve source. Ordered
`t0 < t2 < t3` timing uses the QVDF reconstruction. Unordered or collapsed
timing uses that row's `spd_mph_*` series, which is TAPLite's reprojected
time-dependent result. Converted `link.csv` boundaries are not used.

## 2. Time periods

The half-open period definitions are:

| period | range |
|---|---|
| AM | [06:00, 09:00) |
| MD | [09:00, 15:00) |
| PM | [15:00, 19:00) |

## 3. Assignment D/C

Assignment demand/capacity is:

\[
DC_{\mathrm{assignment}}
=
\frac{\texttt{volume}}
{\texttt{link\_capacity}\,H_p},
\qquad \texttt{link\_capacity}>0.
\]

Here, `link_capacity` is hourly total-link capacity and \(H_p\) is the full
period duration: 3 hours for AM, 6 hours for MD, and 4 hours for PM. The
recomputed ratio is audited against TAPLite's `doc` field.

The assignment projection retains TAPLite's own `P`, `t0`, `t2`, `t3`, and
`vt2`. CBI calibration does not replace missing or degenerate TAPLite timing.

## 4. Asymmetric QVDF queue shape

For any valid episode:

\[
P=t_3-t_0,\qquad
x=\operatorname{clip}\left(\frac{t-t_0}{P},0,1\right),
\]

\[
\rho
=
\operatorname{clip}\left(\frac{t_2-t_0}{P},10^{-6},1-10^{-6}\right).
\]

The unit-peak queue shape is:

\[
S(x)
=
\left(\frac{x}{\rho}\right)^{4\rho}
\left(\frac{1-x}{1-\rho}\right)^{4(1-\rho)}
\quad \text{for }0<x<1.
\]

The implementation explicitly sets:

\[
S(0)=0,\qquad S(\rho)=1,\qquad S(1)=0.
\]

This replaces the former piecewise-smoothstep shape inside the congested
region. It preserves the detected or assigned `t2`; it never assumes `P/2`.

## 5. Queue magnitude and speed conversion

Let:

- \(v_f\) be free-flow speed;
- \(v_{co}\) be congestion cutoff speed;
- \(v_{t2}\) be minimum speed;
- \(L\) be link length; and
- \(\mu\) be discharge flow per lane.

The protected values are:

\[
v_f^{*}=\max(v_f,v_{co}),\quad
v_{t2}^{*}=\operatorname{clip}(v_{t2},1,v_{co}),
\]

\[
L^{*}=\max(L,10^{-6}),\quad
\mu^{*}=\max(\mu,10^{-6}).
\]

Running time at the cutoff speed is:

\[
R_{co}=\frac{L^{*}}{v_{co}}.
\]

The peak queue needed to reproduce \(v_{t2}^{*}\) is:

\[
Q_{\max}
=
\max
\left[
\mu^{*}
\left(
\frac{L^{*}}{v_{t2}^{*}}-R_{co}
\right),
0
\right].
\]

Inside the episode:

\[
Q(t)=Q_{\max}S(x),
\]

\[
v(t)
=
\frac{L^{*}}
{Q(t)/\mu^{*}+R_{co}}.
\]

Consequently:

\[
v(t_0)=v_{co},\qquad
v(t_2)=v_{t2}^{*},\qquad
v(t_3)=v_{co}.
\]

## 6. Smoothstep outside `t0` and `t3`

Define:

\[
H(z)=z^2(3-2z),\qquad 0\leq z\leq1.
\]

For reconstruction window \([w_0,w_1]\), left and right outer anchor speeds
\(v_L\) and \(v_R\), and episode boundaries \(t_0,t_3\):

\[
v(t)
=
v_L+(v_{co}-v_L)
H\left(\frac{t-w_0}{t_0-w_0}\right),
\quad w_0\leq t<t_0,
\]

\[
v(t)
=
v_{co}+(v_R-v_{co})
H\left(\frac{t-t_3}{w_1-t_3}\right),
\quad t_3<t\leq w_1.
\]

This is the approved outside-boundary behavior for both the CBI and
assignment curves. Smoothstep is not used for the queue shape between `t0`
and `t3`.

## 7. Accepted-episode logic

All accepted, non-overlapping episodes in a link-period are used. Candidate
priority for resolving overlaps is:

1. lowest minimum speed;
2. longest episode duration;
3. earliest `t2`; and
4. original stable source order.

A lower-priority candidate is excluded only if its open interval overlaps an
already selected higher-priority episode. The selection and reason are written
to:

```text
07-reconstruction-and-handoff/
average_weekday_reconstruction_episode_selection.csv
```

For the CBI curve, smoothstep interpolation supplies the uncongested portions
and gaps; each selected episode replaces only its own `t0`-`t3` segment with
the QVDF queue-to-speed reconstruction above.

## 8. No accepted episode

Absence of an accepted CBI episode does not prevent an assignment projection.
When valid assignment `t0`, `t2`, `t3`, and `vt2` exist:

- the CBI comparison curve uses smoothstep between free-flow, cutoff, `vt2`,
  cutoff, and free-flow anchors; and
- the assignment curve uses the QVDF queue shape inside `t0`-`t3` plus the
  smoothstep shoulders outside those boundaries.

If the TAPLite boundaries are unordered or collapsed, the assignment curve
uses TAPLite's `spd_mph_*` samples. A collapsed, zero-volume row therefore
plots the free-flow speeds emitted by TAPLite. If neither ordered boundaries
nor a speed profile exists, the TMC-period is labeled `no_assignment_curve`;
no synthetic episode is invented.

## 9. Availability states

One TMC-period has one of four states:

| status | meaning |
|---|---|
| `ready` | canonical mapping, assignment row, and either ordered boundaries or a TAPLite speed profile exist |
| `unmapped` | no canonical network-link mapping |
| `no_assignment` | mapped link has no assignment result |
| `no_assignment_curve` | assignment exists but neither ordered `t0`/`t2`/`t3`/`vt2` nor a TAPLite `spd_mph_*` profile is usable |

Calibration is used for accepted-episode diagnostics. It does not synthesize
or replace TAPLite assignment timing.

## 10. Code correspondence

- `network_mapping.py`: canonical TMC-to-link pairs and deterministic ranks.
- `reconstruction.py`: QVDF shape, queue-to-speed conversion, smoothstep
  shoulders, multi-episode selection, and full-day reconstruction.
- `assignment.py`: TAPLite link-performance normalization.
- `analysis.py`: link-period assembly, field precedence, boundaries, and
  availability states.
- `metrics.py`: CBI and assignment curves and validation metrics.
- `integrated_dashboard/pipeline.py`: hidden projection staging and final
  single-dashboard assembly.
