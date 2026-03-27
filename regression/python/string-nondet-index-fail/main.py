s = nondet_string(5)
c = s[0]
assert c == "h"  # FALHA - valor não determinístico
