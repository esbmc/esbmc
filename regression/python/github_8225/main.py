# Issue #8225: divmod on a non-numeric operand raises TypeError instead of
# aborting the frontend.
def main():
    try:
        q, r = divmod([], 9)
        assert False
    except TypeError:
        pass
    try:
        divmod("s", 9)
        assert False
    except TypeError:
        pass
    try:
        divmod(9, [])
        assert False
    except TypeError:
        pass
    assert divmod(7, 2) == (3, 1)


main()
