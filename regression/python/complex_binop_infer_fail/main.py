# Negative variant of complex_binop_infer: `3 + 4j` is correctly typed complex,
# but the assertion on its real part does not hold. Confirms the inferred
# complex value flows precisely (a wrong claim is refuted, not vacuous).

z = 3 + 4j
assert z.real == 99.0
