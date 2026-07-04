# str.format() thousands grouping "{:,}" / "{:,d}" now folds for a constant
# non-negative integer, including with a space-fill width and an explicit
# sign. Previously the ',' was rejected and the spec fell to a nondet
# fallback. Grouping combined with '0'-fill width (where CPython groups the
# pad zeros) stays on the nondet path, so no wrong value is produced.
assert "{:,}".format(1000000) == "1,000,000"
assert "{:,d}".format(1234567) == "1,234,567"
assert "{:,}".format(1000) == "1,000"
assert "{:,}".format(100) == "100"
assert "{:,}".format(0) == "0"
assert "{:>12,}".format(1000) == "       1,000"
assert "{:+,}".format(1000) == "+1,000"
