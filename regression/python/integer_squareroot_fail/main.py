<<<<<<< HEAD
<<<<<<< HEAD
def integer_squareroot(n: int) -> int:
=======
def integer_squareroot(n: uint64) -> uint64:
>>>>>>> e2b480c2a ([numpy] added integer_squareroot test case)
=======
def integer_squareroot(n: int) -> int:
>>>>>>> 5b7dded0b ([numpy] added integer_squareroot test case)
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 5b7dded0b ([numpy] added integer_squareroot test case)

def test_integer_squareroot():
    # Edge case: smallest input
    assert integer_squareroot(0) == 0

    # Small values
    assert integer_squareroot(1) == 1
    assert integer_squareroot(2) == 1
    assert integer_squareroot(3) == 1
    assert integer_squareroot(4) == 2
    assert integer_squareroot(5) == 2
    assert integer_squareroot(8) == 2
    assert integer_squareroot(9) == 3

    # Very large numbers
    assert integer_squareroot((1 << 64) - 1) == (1 << 32) - 1  # sqrt(2^64 - 1)
    assert integer_squareroot((1 << 62)) == 1 << 31            # sqrt(2^62) = 2^31

    # Large perfect square
    big = (1 << 32)
    assert integer_squareroot(big * big) == big

test_integer_squareroot()
<<<<<<< HEAD
=======
>>>>>>> e2b480c2a ([numpy] added integer_squareroot test case)
=======
>>>>>>> 5b7dded0b ([numpy] added integer_squareroot test case)
