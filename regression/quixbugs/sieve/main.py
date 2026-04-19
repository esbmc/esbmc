
def sieve(max):
    primes = []
    for n in range(2, max + 1):
        if all(n % p > 0 for p in primes):
            primes.append(n)
    return primes

"""
def sieve(max):
    primes = []
    for n in range(2, max + 1):
        if not any(n % p == 0 for p in primes):
            primes.append(n)
    return primes

def sieve(max):
    primes = []
    for n in range(2, max + 1):
        if all(n % p for p in primes):
            primes.append(n)
    return primes

def sieve(max):
    primes = []
    for n in range(2, max + 1):
        if not any(n % p for p in primes):
            primes.append(n)
    return primes

"""

#assert sieve(1) == []
#assert sieve(2) == [2]
assert sieve(4) == [2, 3]
#assert sieve(7) == [2, 3, 5, 7]
#assert sieve(20) == [2, 3, 5, 7, 11, 13, 17, 19]
assert sieve(50) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

