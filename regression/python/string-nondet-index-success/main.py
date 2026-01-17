
s = nondet_string(5)
assume(s == "hello")
assert s[0] == "h"
assert s[1] == "e"
assert s[-1] == "o"
