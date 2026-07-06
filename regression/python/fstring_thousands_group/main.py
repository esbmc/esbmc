# f-string thousands grouping "{:,}" / "{:,d}" now folds for a constant
# integer. Previously the ',' was not consumed and the whole spec fell to a
# nondet-string fallback (sound but imprecise). Covers positive, negative,
# small (no separator), variable, and the ',d' form.
assert f"{1000000:,}" == "1,000,000"
assert f"{1234567:,d}" == "1,234,567"
assert f"{1000:,}" == "1,000"
assert f"{100:,}" == "100"
assert f"{0:,}" == "0"
assert f"{-12345:,}" == "-12,345"
n = 9876543
assert f"{n:,}" == "9,876,543"
