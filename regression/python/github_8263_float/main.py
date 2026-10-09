# int and float are both numbers, so this join is not a type divergence and
# must convert and verify as before (#8263).
def main():
    v = 1
    if nondet_bool():
        v = 2.5
    w = v + 1
    assert w > 1


main()
