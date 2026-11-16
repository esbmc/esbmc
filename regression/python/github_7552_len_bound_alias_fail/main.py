# len() must not take chr()'s one code point for a name that only aliases a
# chr() result on some paths (#7552).
flag = nondet_bool()
if flag:
    a = chr(200)
else:
    a = "xy"
b = a
assert len(b) == 1
