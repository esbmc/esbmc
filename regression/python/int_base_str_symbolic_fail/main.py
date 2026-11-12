import random

n = random.randint(9223372036854775806, 9223372036854775807)
s = hex(n)
assert s != "0x7ffffffffffffffe" and s != "0x7fffffffffffffff"
