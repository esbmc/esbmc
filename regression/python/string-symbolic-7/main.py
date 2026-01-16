
s = nondet_string(5)
assume(s == "hello")
first_char = s[0]
assert first_char == "h"
last_two = s[3:5]
assert last_two == "lo"
