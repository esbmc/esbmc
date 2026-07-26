# KNOWNBUG: the set-relation methods issubset/issuperset/isdisjoint give the
# wrong answer when the RECEIVER is an inline set literal (e.g. {1, 2}.issubset(
# ...)), rather than a set variable. On a variable receiver they are correct:
#
#     a = {1, 2}; b = {1, 2, 3}; a.issubset(b)   -> True  (correct)
#     {1, 2}.issubset({1, 2, 3})                 -> False (WRONG; CPython True)
#
# The inline-literal receiver temp is not materialised the way a named set is
# (the same inline-instance limitation as other `<literal>.method()` receivers),
# so the relation loop sees an empty/mismatched set. The variable form and the
# hand-written `for x in a: x in b` loop both work.
assert {1, 2}.issubset({1, 2, 3})
