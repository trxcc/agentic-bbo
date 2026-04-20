# Domain Prior Knowledge

- All ten controls behave like nonnegative dosage or composition knobs, so interior exploration is usually easier to interpret than repeatedly probing box corners.
- `P10-MIX1` is the only exposed variable whose staged data stay noticeably away from zero; changing it often has larger geometric effect than toggling trace-level additives.
- The benchmark reports transformed regret rather than raw `Target`, but the optimization preference is unchanged: minimizing regret is equivalent to maximizing the underlying predicted target.
- No stronger mechanistic monotonicity is assumed beyond what is present in the staged data and the fitted surrogate.
