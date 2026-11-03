# A caught AssertionError does not end the path, so the read after the
# handler is still checked under --python-ignore-assertions.
def main():
    l = [0, 1]
    try:
        raise AssertionError
    except AssertionError:
        pass
    x = l[2]


main()
