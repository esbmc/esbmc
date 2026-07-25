# Negative counterpart: same non-boolean `or` ternary under the hop-off path,
# but with a false assertion. The sole-adjuster path must still lower the IF and
# reach the solver to report the violation, not crash on the condition type.
def pick_len(s: str, t: str) -> int:
    return len(s) or len(t)


assert pick_len("", "abc") == 99
