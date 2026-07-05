# Negative variant of tuple_ordering: a lexicographic comparison that does not
# hold. Confirms the element-wise lowering decides the comparison precisely
# (a false claim is refuted, not vacuously accepted).

assert (3, 4) < (1, 2)
