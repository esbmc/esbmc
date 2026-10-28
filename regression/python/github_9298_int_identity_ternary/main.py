flag = nondet_bool()
a = int("1001")
b = int("1001")
first = a if flag else "text"
second = b if flag else "other"
assert not (first is second)
