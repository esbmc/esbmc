flag = nondet_bool()
n = nondet_int()
if flag:
    first = n
else:
    first = "text"
if flag:
    second = n
else:
    second = "text"
same = first is second
assert same or not same
