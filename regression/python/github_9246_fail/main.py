# A str key stored with d[s] = v is iterated as a str (#9246).
s = nondet_str()
t = nondet_str()
keys = [t]
d = {}
d[s] = 1
for k in d:
    assert k in keys
