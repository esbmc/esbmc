# Test nondet_str() in conditional with multiple comparisons

s = nondet_str()

# Use ESBMC assume to constrain the nondeterministic value
__ESBMC_assume(s != "")

# After assume, s cannot be empty
assert len(s) > 0

