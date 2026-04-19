def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a % b)


assert gcd(17, 0) == 17
assert gcd(13, 13) == 13
assert gcd(37, 600) == 1
assert gcd(20, 100) == 20
assert gcd(624129, 2061517) == 18913
assert gcd(3, 12) == 3

