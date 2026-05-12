counter = 0


def nxt():
    global counter
    counter += 1
    return counter


def main():
    # Four-pivot chain — exercises the loop body more than once; each
    # inner pivot must still be evaluated exactly once (counter == 4).
    if nxt() < nxt() < nxt() < nxt():
        pass
    assert counter == 4


main()
