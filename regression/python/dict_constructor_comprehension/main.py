# dict() over a comprehension of (key, value) pairs is lowered to an empty
# dict plus population loops (HumanEval/126). Exercise the list-comprehension
# form, the generator-expression form, and a filtered comprehension.


def main() -> None:
    squares = dict([(i, i * i) for i in [1, 2, 3]])
    assert squares[3] == 9

    doubled = dict((k, k + k) for k in [5, 6])
    assert doubled[6] == 12

    evens = dict([(n, n) for n in [1, 2, 3] if n % 2 == 0])
    assert evens[2] == 2


main()
