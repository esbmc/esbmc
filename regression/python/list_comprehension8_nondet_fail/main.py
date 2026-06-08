def q(a):
    return a[0]


def main():
    # Companion to list_comprehension8_nondet: the element-type recovery must
    # not over-constrain comprehension elements into something trivially true.
    # A nondet element read through a parameter alias can be any int, so
    # asserting it equals a fixed constant is genuinely violable.
    nums = [nondet_int() for _ in range(2)]
    assert q(nums) == 7


main()
