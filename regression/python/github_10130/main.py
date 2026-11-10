# Issue #10130: a str-and-bool and/or tested inside another and/or
t = ""
n = 0
if (t and t == "a") or n == 1:
    n += 5
assert n == 0

k = 0
while (t and t == "a") or k < 1:
    k += 1
assert k == 1

x = (t and t == "a") or n == 1
assert not x

u = "a"
if (u and u == "a") or n == 1:
    n += 5
assert n == 5

assert not (t and t == "a")
