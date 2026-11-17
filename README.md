# GTA VI launch-day Twitch CCU forecast — v4

Forecast GTA VI's launch-day Twitch viewers hour by hour (hours 2–24
UTC, 19 Nov 2026), as five quantiles (5/25/50/75/95%) of concurrent
viewers. Entry for make.fun competition
`gta6-twitch-ccu-forecast-01`. Pure Python 3 stdlib, offline,
deterministic, ~0.1 s per case.

## Model

Two-regime log-ratio forecaster. Each comparable launch contributes
its hour-1 viewer count and, per later hour, the log ratio of that
hour's viewers to hour 1. For each hour a size-weighted least-squares
fit predicts the log ratio from the target's hour-1 size, with a
one-sided quadratic size term, a Thursday level + Thursday × size
interaction, and Friday-only weekday matching (×4 when the target
day is Friday). The median is shrunk 20% toward the pooled median
curve; quantiles come from the fit's residual sigma.

The v4 change is that the model switches with launch size, because
the final is a gigantic Thursday launch and giants behave
differently from the pack:

- **SMALL regime** (hour-1 < 300,000): v3's recipe plus a
  viewers-per-channel feature (hour-1 viewers ÷ hour-1 live
  channels, coefficient halved) and dummy smoothing across
  neighbouring hours. The channels feature is the single biggest
  practice win in testing.
- **GIANT regime** (hour-1 ≥ 300,000): quadratic size weights
  (giants dominate the fit), narrower spread (0.35), dummy
  smoothing, and — the key change — **empirical quantiles** from
  the pooled standardised residuals instead of normal z-scores.

## Validated performance (main.py run end-to-end)

| cut | v4 | v3 |
| --- | --- | --- |
| practice | **0.1447** | 0.1527 |
| LOO-big (>30k) | **0.2302** | 0.2352 |
| LOO-Thu-big | **0.1430** | 0.1552 |
| LOO-Thu-top (>100k Thu) | **0.1270** | 0.1410 |
| LOO-all | **0.4640** | 0.4664 |

Beats v3 on every cut. On the closest analogue to the final,
Cyberpunk 2077 (852k hour-1, Thursday): 0.0721 vs v3's 0.1074.

## Run

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

Reads every `*.in` in the input dir, writes one `<id>.out` per case:
23 rows (hours 2–24), five monotone quantiles in (0, 1e9].

## Files

- `main.py` — the whole forecaster (stdlib only)
- `LICENSE` — MIT
