from esbmc import nondet_bool

c = nondet_bool()
s = "xy" if c else "z"
assert s == ("xy" if c else "z")
assert s != ("q" if c else "w")
