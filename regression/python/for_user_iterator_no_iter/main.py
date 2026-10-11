# A class with __next__ but no __iter__ is not iterable, and the TypeError
# can be caught.
class N:
    def __init__(self):
        self.i = 0

    def __next__(self) -> int:
        raise StopIteration


def main():
    caught = False
    try:
        for x in N():
            pass
    except TypeError:
        caught = True
    assert caught


main()
