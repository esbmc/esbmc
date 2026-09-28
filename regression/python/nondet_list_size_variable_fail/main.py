# Companion to nondet_list_size_variable: the bound really is the variable's
# value, so a list of 5 elements is reachable and a bound of 4 is falsified.
nondet_size = 5
x = nondet_list(nondet_size)
assert len(x) <= 4
