# Negative variant of dictcomp_tuple_target: the tuple-target dict comprehension
# is built correctly, but the assertion does not hold. Confirms the unpacked
# values flow precisely (a wrong claim is refuted, not vacuously accepted).

pairs = [(1, 2), (3, 4)]

d = {v: u for u, v in pairs}
assert d[2] == 99
