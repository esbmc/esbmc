# Guards against over-poisoning: at --unwind 1 only a successful conversion-
# time fold can verify the count() assert (the operational model searches
# with a loop that would truncate). A string global through a pure function
# must also still seed and fold. len() is loop-free either way and is kept
# only as a smoke check.
S = "abcdefabcdef"
def mid(s: str) -> str:
    return s[1:5]
assert mid(S) == "bcde"

L = [1, 2, 3]
assert L.count(2) == 1
assert len(L) == 3
