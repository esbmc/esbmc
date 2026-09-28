# Issue #7769: the per-site partition must stay complete. ValueError has two
# attributable raise sites (lines 8 and 10). The raise on line 12 is caught where
# it is raised, so it gets no site of its own, and the bare `raise` re-raises it
# past both sites' properties. Only the residual property can catch that
# escape; without it this program would verify.
def check(x: int) -> None:
    if x > 10:
        raise ValueError("large")
    if x > 5:
        raise ValueError("medium")
    try:
        raise ValueError("small")
    except ValueError:
        raise


check(0)
