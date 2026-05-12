counter = 0


def nxt():
    global counter
    counter += 1
    return counter


def main():
    if nxt() < nxt() < nxt():
        pass
    # counter is 3 (single eval); the assertion below must fail.
    assert counter == 4


main()
