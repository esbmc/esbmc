# A misspelled keyword used to be dropped in silence, reverting the element type
# to int and proving float properties vacuously. It is now rejected by name.
x = nondet_list(3, elemtype=nondet_float())
assert len(x) >= 0
