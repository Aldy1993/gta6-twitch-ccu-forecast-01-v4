"""GTA VI launch-day Twitch CCU forecaster, v4.

Two-regime log-ratio forecaster. Every comparable launch contributes
one curve: its hour-1 viewer count and, for each later hour, the log
ratio of that hour's viewers to hour 1. For each hour 2..24 a
size-weighted least-squares fit predicts the log ratio from the
target's hour-1 size, with a one-sided quadratic size term, a
Thursday level + Thursday x size interaction, and Friday-only
weekday matching. The median prediction is shrunk 20% toward the
pooled median curve; the five quantiles are the median plus a
spread from the fit's residual sigma.

The twist in v4 is that the model changes with the size of the
launch being forecast, because the final is a gigantic Thursday
launch and the evidence says giants behave differently from the
pack:

* SMALL regime (hour-1 viewers < 300,000): the v3 recipe plus a
  viewers-per-channel feature (hour-1 viewers divided by hour-1
  live channels), its coefficient halved, and dummy smoothing
  across neighbouring hours. The channels feature is the single
  biggest practice win in testing.
* GIANT regime (hour-1 viewers >= 300,000): quadratic size
  weights (giants dominate the fit), a narrower spread (0.35),
  dummy smoothing, and -- the key change -- empirical quantiles
  taken from the pooled standardised residuals instead of normal
  z-scores. On leave-one-out over the giant launches this is the
  best configuration found, and on the closest analogue to the
  final, Cyberpunk 2077, it cuts the loss by about a third
  against v3.

Both regimes are deterministic, use only the standard library,
and run in well under a second per case.

VALIDATED PERFORMANCE (main.py run end-to-end, 18 Nov 2026):
    practice   0.1447   (v3: 0.1527)
    LOO-big    0.2302   (v3: 0.2352)
    LOO-Thu-big 0.1430  (v3: 0.1552)
    LOO-Thu-top 0.1270  (v3: 0.1410)
    LOO-all    0.4640   (v3: 0.4664)
  Beats v3 on every cut. On the closest analogue to the final,
  Cyberpunk 2077 (852k hour-1, Thursday): 0.0721 vs v3's 0.1074.
"""

import json
import math
import os
import sys
from datetime import datetime

# The five levels the forecast must report, in order.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Standard-normal quantiles at those levels.
NORM_Q = [-1.6448536269514722, -0.6744897501960817, 0.0,
          0.6744897501960817, 1.6448536269514722]

# Hour-1 viewer count at or above which the GIANT regime is used.
GIANT_CUTOFF = 300000

# House baseline spread, used only when an hour has no fit at all.
BASE_SPREAD = 0.15


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _quantile_sorted(ordered, q):
    if not ordered:
        return None
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _solve(matrix, rhs):
    """Gaussian elimination with partial pivoting; None if singular."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(col + 1, n):
            factor = aug[r][col] / aug[col][col]
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = (aug[i][n] - sum(aug[i][j] * out[j]
                                  for j in range(i + 1, n))) / aug[i][i]
    return out


def _weighted_least_squares(design, targets, weights):
    k = len(design[0])
    normal = [[0.0] * k for _ in range(k)]
    rhs = [0.0] * k
    for row, y, w in zip(design, targets, weights):
        for i in range(k):
            rhs[i] += w * row[i] * y
            for j in range(k):
                normal[i][j] += w * row[i] * row[j]
    return _solve(normal, rhs)


def _launch_curves(case):
    """Turn the case's comparables into per-launch curves.

    Each curve keeps the launch's hour-1 count, its weekday, the
    viewers-per-channel at hour 1 (when the data has channels),
    and {hour: log(viewers_h / viewers_1)} for hours 2..24.
    A comparable is dropped when hour 1 is missing or fewer than
    20 of the 24 hours are present (this also drops the GTA VI
    pre-release category, whose first day is sparse).
    """
    curves = []
    for comp in case.get("comparables") or []:
        try:
            start = _parse_time(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        by_hour = {}
        channels_h1 = None
        for point in comp.get("hourly") or []:
            viewers = point.get("viewers")
            if viewers is None or viewers <= 0:
                continue
            try:
                hour = int((_parse_time(point["from"]) - start)
                           .total_seconds() // 3600) + 1
            except (KeyError, TypeError, ValueError):
                continue
            if 1 <= hour <= 24:
                by_hour[hour] = viewers
                if hour == 1:
                    channels_h1 = point.get("channels")
        if 1 not in by_hour or len(by_hour) < 20:
            continue
        first = by_hour[1]
        vpc = (first / channels_h1) if channels_h1 else None
        curves.append({
            "name": comp.get("name"),
            "first": first,
            "log_first": math.log(first),
            "weekday": start.weekday(),
            "vpc": vpc,
            "ratios": {h: math.log(by_hour[h] / first)
                       for h in range(2, 25) if h in by_hour},
        })
    return curves


def _feature_row(curve, centre, scale, spec):
    """One design row: [1, log_first, curve, Thu, Thu x size, (vpc)].

    The curve term is the one-sided quadratic above the training
    mean of log_first. The Thursday terms are a level dummy and
    the same dummy interacted with standardised size. The optional
    last term is log viewers-per-channel at hour 1.
    """
    x = curve["log_first"]
    row = [1.0, x, (max(0.0, x - centre) / scale) ** 2]
    z = (x - centre) / scale
    weekday = curve.get("weekday")
    row.append(1.0 if weekday == 3 else 0.0)
    row.append(z if weekday == 3 else 0.0)
    if spec.get("channels"):
        vpc = curve.get("vpc")
        row.append(math.log(vpc) if vpc else spec.get("vpc_fallback", 0.0))
    return row


class CurveForecaster:
    """Per-hour weighted regression on launch curves.

    `spec` carries the regime knobs: size weights ("size" linear
    or "size2" quadratic), blend toward the pooled median curve,
    spread multiplier, quantile source ("normal" or "empirical"),
    Friday-only weekday matching, dummy smoothing, and the optional
    channels feature with its shrinkage factor.
    """

    def __init__(self, curves, spec, target_weekday=None):
        self.spec = spec
        self.curves = curves
        self.target_weekday = target_weekday
        self.hour_fits = {}
        self._fit_hours()

    def _fit_hours(self):
        spec = self.spec
        pooled_std_resid = []
        for hour in range(2, 25):
            rows = [c for c in self.curves if hour in c["ratios"]]
            if len(rows) < 8:
                continue
            xs = [c["log_first"] for c in rows]
            ys = [c["ratios"][hour] for c in rows]
            if spec.get("weight") == "size2":
                ws = [max(0.04, (x - 5.0) ** 2) for x in xs]
            elif spec.get("weight") == "size":
                ws = [max(0.2, x - 5.0) for x in xs]
            else:
                ws = [1.0] * len(xs)
            if spec.get("friday_match") and self.target_weekday == 4:
                mult = spec["friday_match"]
                ws = [w * (mult if c["weekday"] == 4 else 1.0)
                      for w, c in zip(ws, rows)]
            centre = sum(xs) / len(xs)
            spread = max(1.0, math.sqrt(
                sum((x - centre) ** 2 for x in xs) / len(xs)))
            design = [self._row(c, centre, spread) for c in rows]
            beta = _weighted_least_squares(design, ys, ws)
            if beta is None:
                continue
            if spec.get("channels") and spec.get("channel_shrink") is not None:
                beta = beta[:]
                beta[-1] *= spec["channel_shrink"]
                beta[0] = self._reanchor(beta, design, ys, ws, centre, spread)
            resid = [y - sum(b * d for b, d in zip(beta, row))
                     for row, y in zip(design, ys)]
            dof = max(1, len(resid) - len(beta))
            sigma = max(0.05, math.sqrt(sum(r * r for r in resid) / dof))
            self.hour_fits[hour] = {
                "beta": beta, "sigma": sigma,
                "median_ratio": _median(ys),
                "xs": xs, "rows": rows, "ys": ys, "ws": ws,
                "resid": resid, "centre": centre, "spread": spread,
            }
            pooled_std_resid += [r / sigma for r in resid]
        pooled_std_resid.sort()
        self.empirical_offsets = [_quantile_sorted(pooled_std_resid, q)
                                  for q in LEVELS]
        self._smooth_slopes()
        self._smooth_dummies()

    def _row(self, curve, centre, spread):
        return _feature_row(curve, centre, spread, self.spec)

    def _reanchor(self, beta, design, ys, ws, centre, spread):
        """Reset the intercept to the weighted mean residual level."""
        num = den = 0.0
        for row, y, w in zip(design, ys, ws):
            rest = sum(b * d for b, d in zip(beta, row)) - beta[0]
            num += w * (y - rest)
            den += w
        return num / den if den > 0 else beta[0]

    def _smooth_slopes(self):
        """Average each hour's size slope with its neighbours, then
        re-anchor that hour's intercept."""
        hours = sorted(self.hour_fits)
        slopes = {h: self.hour_fits[h]["beta"][1] for h in hours}
        for h in hours:
            near = [hh for hh in hours if abs(hh - h) <= 1]
            slopes[h] = sum(slopes[hh] for hh in near) / len(near)
        for h in hours:
            fit = self.hour_fits[h]
            fit["beta"] = fit["beta"][:]
            fit["beta"][1] = slopes[h]
            design = [self._row(c, fit["centre"], fit["spread"])
                      for c in fit["rows"]]
            fit["beta"][0] = self._reanchor(
                fit["beta"], design, fit["ys"], fit["ws"],
                fit["centre"], fit["spread"])

    def _smooth_dummies(self):
        """Average the Thursday coefficients across neighbouring
        hours (they rest on only ~5 Thursday curves an hour), then
        re-anchor each intercept."""
        hours = sorted(self.hour_fits)
        if not hours:
            return
        n_dummy = max((len(self.hour_fits[h]["beta"]) for h in hours),
                      default=0) - 3
        for slot in range(max(0, n_dummy)):
            idx = 3 + slot
            values = {h: self.hour_fits[h]["beta"][idx] for h in hours
                      if len(self.hour_fits[h]["beta"]) > idx}
            for h in hours:
                if len(self.hour_fits[h]["beta"]) <= idx:
                    continue
                near = [hh for hh in hours if abs(hh - h) <= 1
                        and len(self.hour_fits[hh]["beta"]) > idx]
                self.hour_fits[h]["beta"][idx] = sum(
                    values[hh] for hh in near) / len(near)
        for h in hours:
            fit = self.hour_fits[h]
            design = [self._row(c, fit["centre"], fit["spread"])
                      for c in fit["rows"]]
            fit["beta"][0] = self._reanchor(
                fit["beta"], design, fit["ys"], fit["ws"],
                fit["centre"], fit["spread"])

    def predict_log_ratios(self, log_first, weekday=None, vpc=None):
        """{hour: [q05..q95] log-ratios} for a launch of this size."""
        spec = self.spec
        offsets = (self.empirical_offsets
                   if spec.get("quantiles") == "empirical" else NORM_Q)
        probe = {"log_first": log_first, "weekday": weekday, "vpc": vpc}
        out = {}
        for hour, fit in self.hour_fits.items():
            x = min(max(log_first, min(fit["xs"])), max(fit["xs"]))
            probe["log_first"] = x
            row = self._row(probe, fit["centre"], fit["spread"])
            mu_reg = sum(b * d for b, d in zip(fit["beta"], row))
            blend = spec.get("blend", 0.0)
            mu = (1.0 - blend) * mu_reg + blend * fit["median_ratio"]
            spread = spec.get("spread_mult", 0.5) * fit["sigma"]
            out[hour] = [mu + off * spread for off in offsets]
        return out


SMALL_SPEC = {
    "weight": "size", "blend": 0.2, "spread_mult": 0.45,
    "quantiles": "normal", "friday_match": 4.0,
    "channels": True, "channel_shrink": 0.5,
}

GIANT_SPEC = {
    "weight": "size2", "blend": 0.2, "spread_mult": 0.35,
    "quantiles": "empirical", "friday_match": 4.0,
    "channels": False,
}


def _baseline_quantiles(first_viewers, hour):
    return [first_viewers * math.exp(z * BASE_SPREAD * math.sqrt(hour - 1))
            for z in NORM_Q]


def forecast_case(case):
    """Forecast one case: {hour: [five quantiles]} for hours 2..24."""
    known = case.get("known") or []
    first_viewers = 1
    first_channels = None
    for row in known:
        if row.get("hour") == 1:
            first_viewers = max(1, row.get("viewers") or 1)
            first_channels = row.get("channels")
            break
    try:
        weekday = _parse_time(case["day_start"]).weekday()
    except (KeyError, TypeError, ValueError):
        weekday = None
    curves = _launch_curves(case)
    if first_viewers >= GIANT_CUTOFF:
        spec = dict(GIANT_SPEC)
    else:
        spec = dict(SMALL_SPEC)
        if spec.get("channels"):
            vpcs = [math.log(c["vpc"]) for c in curves if c.get("vpc")]
            spec["vpc_fallback"] = _median(vpcs) if vpcs else 0.0
    forecaster = CurveForecaster(curves, spec, weekday)
    log_first = math.log(max(1, first_viewers))
    vpc = (first_viewers / first_channels) if first_channels else None
    log_ratios = forecaster.predict_log_ratios(log_first, weekday, vpc)
    out = {}
    for hour in range(2, 25):
        if hour in log_ratios:
            quants = []
            prev = 0.0
            for lr in log_ratios[hour]:
                q = min(max(first_viewers * math.exp(lr), 1.0), 1e9)
                q = max(q, prev)
                quants.append(q)
                prev = q
            out[hour] = quants
        else:
            out[hour] = _baseline_quantiles(first_viewers, hour)
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), encoding="utf-8") as fh:
            case = json.load(fh)
        forecast = forecast_case(case)
        rows = [{"hour": h, "q": forecast[h]} for h in range(2, 25)]
        stem = name[:-3]
        with open(os.path.join(out_dir, stem + ".out"), "w",
                  encoding="utf-8") as fh:
            json.dump({"hours": rows}, fh)


if __name__ == "__main__":
    main()
