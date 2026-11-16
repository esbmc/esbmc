import random
import p.s

m = p.s if random.random() < 0.5 else p
r = m.func()
assert r == 1 or r == 2
