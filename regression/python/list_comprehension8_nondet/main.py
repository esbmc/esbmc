def p(a):
    return a[0] and a[1]


def q(a):
    return a[0]


def main():
    # Issue #5022: a list built by a comprehension and read through a
    # function-parameter alias must preserve its element type. A pure function
    # applied twice to the same unmodified list must return the same value.
    flags = [nondet_bool() for _ in range(2)]
    assert p(flags) == p(flags)

    nums = [nondet_int() for _ in range(2)]
    assert q(nums) == q(nums)


main()
