# Membership over a *named* tuple of strings must use element-wise string
# equality, not a substring search over the tuple's raw bytes (issue #5150).
def hex_key(num):
    primes = ('2', '3', '5', '7', 'B', 'D')
    total = 0
    for i in range(0, len(num)):
        if num[i] in primes:
            total += 1
    return total


def main():
    primes = ('2', '3', '5', '7', 'B', 'D')
    assert 'B' in primes
    assert 'D' in primes
    assert 'A' not in primes
    assert 'C' not in primes
    assert hex_key("AB") == 1


main()
