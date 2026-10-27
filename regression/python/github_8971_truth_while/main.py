flag = nondet_bool()
x = nondet_int() if flag else nondet_str()
n = 0
while x:
    n += 1
    break
assert (n == 1) == ((isinstance(x, int) and x != 0) or (isinstance(x, str) and x != ""))
