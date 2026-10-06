"""One checkpoint-first RSI-style reporting policy for every experiment."""

import math

import numpy as np

VERSION = "rsi_unified_exponential_v2_20260923"
EXP_FAMILIES = frozenset(("HPO", "DBTune", "BBOPlace"))


def tolerance(*values):
    return 1e-12 * max(1., *(abs(float(x)) for x in values if x is not None))


def reference_spec(initial_losses, gp_loss=None, upper_loss=None, *, family):
    values = np.asarray(initial_losses, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid initialization")
    b = -float(min(values))
    r = None if gp_loss is None else -float(gp_loss)
    u = None if upper_loss is None else -float(upper_loss)
    spec = dict(family=family, b=b, r=r, u=u, scale=None, scale_source=None,
                status="valid", mode=None, gp_anchor_used=False)
    if family in EXP_FAMILIES:
        spec["u"] = None
        if r is None or not math.isfinite(r):
            raise ValueError("Missing/nonfinite matched GP")
        if r - b > tolerance(b, r):
            spec.update(mode="gp_linear_smooth_exponential_tail", gp_anchor_used=True,
                        scale=r-b, scale_source="matched_gp_improvement")
        else:
            iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
            span = float(np.ptp(values))
            if iqr > tolerance(*values):
                scale, source = iqr, "shared_initialization_IQR"
            elif span > tolerance(*values):
                scale, source = span, "shared_initialization_range"
            else:
                raise ValueError("Flat initialization: no outcome-independent scale")
            spec.update(mode="no_gp_gain_initialization_exponential", scale=scale, scale_source=source)
        return spec
    if u is None or not math.isfinite(u) or u-b <= tolerance(b, u):
        raise ValueError("Missing bound or initialization already at bound")
    if family == "GuacaMol":
        spec.update(mode="molecular_linear")
    elif r is None or not math.isfinite(r):
        raise ValueError("Missing/nonfinite matched GP")
    elif r-b <= tolerance(b, r, u):
        spec.update(mode="linear_gp_no_improvement")
    elif u-r <= tolerance(b, r, u):
        if r > u + tolerance(b, r, u):
            raise ValueError("GP exceeds theoretical bound")
        # The optional 0.6 anchor cannot occupy the same point as the 1 anchor.
        spec.update(mode="linear_gp_at_upper")
    else:
        spec.update(mode="bounded_gp_piecewise", gp_anchor_used=True)
    return spec


def quality(y, spec):
    b, r, u = (spec[k] for k in ("b", "r", "u"))
    if not math.isfinite(y) or y < b-tolerance(b, y):
        raise ValueError("Invalid incumbent")
    y = max(b, float(y))
    mode = spec["mode"]
    if mode == "gp_linear_smooth_exponential_tail":
        return .6*(y-b)/(r-b) if y <= r else 1-.4*math.exp(-1.5*(y-r)/(r-b))
    if mode == "no_gp_gain_initialization_exponential":
        return -math.expm1(-(y-b)/spec["scale"])
    if mode in ("molecular_linear", "linear_gp_no_improvement", "linear_gp_at_upper"):
        return min(1., (y-b)/(u-b))
    if y <= r:
        return .6*(y-b)/(r-b)
    return min(1., .6+.4*(y-r)/(u-r))


def score_losses(losses, initial, budget, upper_loss=None, gp_loss=None, *, family, early_end=False):
    values = np.asarray(losses, dtype=float)
    if initial < 1 or budget < 1 or not initial <= len(values) <= initial+budget or not np.isfinite(values).all():
        raise ValueError("Invalid trajectory")
    if len(values) < initial+budget and not early_end:
        raise ValueError("Incomplete trajectory without explicit carry-forward")
    spec = reference_spec(values[:initial], gp_loss, upper_loss, family=family)
    best = np.minimum.accumulate(values)
    checkpoints = list(-best[initial:]) + [-float(best[-1])]*(initial+budget-len(values))
    q = [quality(y, spec) for y in checkpoints]
    anytime, final = math.fsum(q)/budget, q[-1]
    result = dict(normalization_status="valid", normalization_mode=spec["mode"],
                  gp_anchor_used=spec["gp_anchor_used"], initial_loss=-spec["b"],
                  gp_reference_loss=gp_loss, upper_reference_loss=None if spec["u"] is None else -spec["u"],
                  exponential_scale=spec["scale"], scale_source=spec["scale_source"],
                  final_loss=float(best[-1]), observed_new=len(values)-initial, budget=budget,
                  filled_checkpoints=initial+budget-len(values), anytime_score=anytime,
                  final_score=final, composite_score=.7*anytime+.3*final)
    return result, q, spec
