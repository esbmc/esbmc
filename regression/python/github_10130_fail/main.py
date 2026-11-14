# Issue #10130: a str-and-bool and/or tested inside another and/or
t = ""
n = 0
if (t and t == "a") or n == 1:
    n += 5
assert n == 0

u = "a"
if (u and u == "a") or n == 1:
    n += 5
assert n == 0
