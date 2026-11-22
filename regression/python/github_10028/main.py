import re

assert re.sub(r"b", "", "abc") == "ac"
assert re.sub("ab", "X", "abcabab") == "XcXX"
assert re.sub("ab", "X", "abcabab", 2) == "XcXab"
assert re.sub("ab", "X", "abcabab", count=1) == "Xcabab"
assert re.sub("", "-", "ab") == "-a-b-"
assert re.sub("", "-", "ab", 2) == "-a-b"
assert re.sub("z", "y", "abc", flags=0) == "abc"

n: int = nondet_int()
__ESBMC_assume(n >= 0 and n <= 3)
assert re.sub("a", "b", "aaaa", n).count("b") == (4 if n == 0 else n)
