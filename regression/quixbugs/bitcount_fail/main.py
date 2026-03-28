
def bitcount(n):
    count = 0
    while n:
        n ^= n - 1
        count += 1
    return count


"""
Bitcount
bitcount


Input:
    n: a nonnegative int

Output:
    The number of 1-bits in the binary encoding of n

Examples:
    >>> bitcount(127)
    7
    >>> bitcount(128)
    1
"""

assert bitcount(127) == 7
assert bitcount(128) == 1
assert bitcount(3005) == 9
assert bitcount(13) == 3
assert bitcount(14) == 3
assert bitcount(27) == 4
assert bitcount(834) == 4
assert bitcount(254) == 7
assert bitcount(256) == 1
