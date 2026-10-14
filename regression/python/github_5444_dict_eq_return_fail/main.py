# Negative twin of github_5444_dict_eq_return.
#
# Same return-across-call-boundary shape (float('inf')-init slot reassigned
# to an int, dict returned, compared against an int-literal dict), but the
# *untouched* slot ("b") now genuinely differs: mk() never changes "b", so
# the returned dict still holds "b": 0, while the comparison literal claims
# "b": 1. CPython raises AssertionError (exit 1); ESBMC must report
# VERIFICATION FAILED. Guards against any future regression that would
# let a real mismatch on an untouched slot go undetected once a different
# slot's numeric int/float fallback fires (dict-equality model,
# list.c:845-939).
def mk(a: int) -> dict:
    d = {"a": float("inf"), "b": 0}
    d["a"] = a
    return d


r = mk(1)
assert r == {"a": 1, "b": 1}
