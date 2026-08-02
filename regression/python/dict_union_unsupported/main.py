# dict union (PEP 584: d1 | d2) is not modeled. Before the fix both dict structs
# fell through to the bitwise BitOr path and SIGSEGV'd in the SMT backend (a
# struct irep handed to bitwuzla_mk_term2 as a term pointer). It must now produce
# a clean "not supported" diagnostic instead of crashing.
a = {"a": 1}
b = {"b": 2}
c = a | b
assert c["b"] == 2
