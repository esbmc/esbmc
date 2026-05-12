counter = 0


def nxt():
    global counter
    counter += 1
    return counter


def main():
    # Per [expressions]/6.10, each pivot of a chained comparison is
    # evaluated *at most once*.  The three calls below should bump
    # counter from 0 to 3, not 4 — i.e. the inner pivot must be shared
    # between the (a < b) and (b < c) arms instead of being re-evaluated.
    if nxt() < nxt() < nxt():
        pass
    assert counter == 3


main()
