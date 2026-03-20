# Issue #3855: reassigning a float variable to chr() result (dynamic type change
# from float to char*) is not modelled — the GOTO IR cannot represent one variable
# holding two different types.  The assignment is silently skipped, so the
# assertion below fails.
a = 60
b = 5
sum = float(a + b)
assert sum == 65.0
sum = chr(int(sum))
assert sum == "A"
