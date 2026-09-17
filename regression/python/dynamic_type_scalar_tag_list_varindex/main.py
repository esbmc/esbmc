# A variable index skips the constant-index block that reads the recorded
# element type, so this read used to unwrap the element as a plain int and
# overran the 2-byte "a" payload.
cond = nondet_bool()
if cond:
    x = 1
else:
    x = "a"
lst = [x]
i = 0
e = lst[i]
assert e == 1 or e == "a"
