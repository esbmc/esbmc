# list.count()/index() on a literal receiver folds at conversion time. The
# OM value-matching path returned a wrong result for a literal (count of a
# present element was 0, index was misresolved), while tuples already worked.
assert [1, 2, 2, 3].count(2) == 2
assert [1, 2, 2, 3].index(2) == 1
assert [1, 2, 3].count(9) == 0
assert ["a", "b", "a"].count("a") == 2
assert ["a", "b", "a"].index("b") == 1
# Tuples and the string form must remain correct too.
assert (1, 2, 2, 3).count(2) == 2
assert "aXbXc".count("X") == 2
