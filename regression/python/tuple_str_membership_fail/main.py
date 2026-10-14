# Negative companion to tuple_str_membership: 'A' is not a prime hex digit, so
# `'A' in primes` is False and the assertion must fail (issue #5150).
def main():
    primes = ('2', '3', '5', '7', 'B', 'D')
    assert 'A' in primes


main()
