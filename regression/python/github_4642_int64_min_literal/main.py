# Python ast.parse splits -9223372036854775808 (= INT64_MIN) into
# UnaryOp(USub, Constant(9223372036854775808)). The inner Constant magnitude
# is technically out of int64 range, but the negated value fits exactly; the
# parser tagger must take that one-level USub context into account and not
# trap. See issue #4642 and PR #4645 (Copilot finding #1).
x = -9223372036854775808
assert x < 0
assert x + 1 < 0
