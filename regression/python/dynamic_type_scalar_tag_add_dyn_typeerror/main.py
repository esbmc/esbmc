# x/y diverge in opposite polarity, so x + y is a num/str mismatch
# either way, an uncaught TypeError regardless of cond.
cond = nondet_bool()
if cond:
    x = 5
    y = "b"
else:
    x = "a"
    y = 3

z = x + y
