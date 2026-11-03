# A str key stored with d[s] = v is iterated as a str (#9246).
s = nondet_str()
keys = [s]
d = {}
d[s] = 1
for k in d:
    assert k in keys
