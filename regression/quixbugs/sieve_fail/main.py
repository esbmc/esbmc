def sieve(max):
    primes = []
    for n in range(2, max + 1):
        if any(n % p > 0 for p in primes):
            primes.append(n)
    return primes


assert sieve(1) == []
assert sieve(2) == [2]
assert sieve(4) == [2, 3]
