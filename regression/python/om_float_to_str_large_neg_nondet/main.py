def nondet_float() -> float: ...

# __python_float_to_str must not cast v > ULLONG_MAX (~1.8e19) to an integer,
# which is UB. str(2e19) is "2e+19", which the model leaves unconstrained apart
# from the 3..24 characters every float repr has.
v = nondet_float()
__ESBMC_assume(v == 2.0e19)
s = str(v)
assert len(s) >= 3
