import random

n = random.randint(-9223372036854775807 - 1, 9223372036854775807)
if n == -9223372036854775807 - 1:
    assert bin(n) == "-0b1000000000000000000000000000000000000000000000000000000000000000"
    assert hex(n) == "-0x8000000000000000"
    assert oct(n) == "-0o1000000000000000000000"
if n == 9223372036854775807:
    assert hex(n) == "0x7fffffffffffffff"
if n == 0:
    assert bin(n) == "0b0"
    assert hex(n) == "0x0"
    assert oct(n) == "0o0"
