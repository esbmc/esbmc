# Sound-twin of github_5444_dict_eq_return_fail.
#
# Regression pin for the "return across the call boundary" shape flagged as
# untested by the #5444 umbrella's DoD note ("sound int/float dict equality
# on returned dicts"): a dict local to a function starts a slot as
# float('inf'), reassigns it to an int via subscript, and is then returned
# and compared (via `==`) against an int-literal dict at the call site.
#
# The dict-equality model's int/float numeric fallback (list.c, added by
# #5726) and the dict's key/value payloads must both survive the
# function-return copy intact -- this is the exact GOTO shape the umbrella
# worried might alias the returned struct's key/value lists with the RHS
# literal's, masking a real mismatch. Confirmed sound by exhaustive
# differential testing against CPython (15 structural variants: key order,
# comparison direction, negated assert, nondet parameter, min()-based
# reassignment, float-vs-int RHS literal); this test pins the true
# (all-equal) case.
def mk(a: int) -> dict:
    d = {"a": float("inf"), "b": 0}
    d["a"] = a
    return d


r = mk(1)
assert r == {"a": 1, "b": 0}
