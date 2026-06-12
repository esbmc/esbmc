# Negative variant of dictcomp_over_items: the items()-based dict comprehension
# is built correctly, but the assertion does not hold. Confirms the unpacked
# (key, value) bindings flow precisely (a wrong claim is refuted).

d = {1: 10, 2: 20}

e = {k: v + 1 for k, v in d.items()}
assert e[1] == 99
