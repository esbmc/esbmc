def sieve(max: int) -> list[int]:
    primes: list[int] = []
    for n in range(2, max + 1):
        if all(n % p > 0 for p in primes):
            primes.append(n)
    return primes


assert sieve(1) == []
assert sieve(2) == [2]
assert sieve(3) == [1, 2, 3]
