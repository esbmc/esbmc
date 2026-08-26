# The IndexError guard on a subscript and the exception-propagation edge after
# a call are frontend instrumentation, not branches of this program: an
# unreachable bounds check is the proof the subscript is safe.
def first(values):
    return values[0]


def main():
    values = [1, 2, 3]
    assert first(values) == 1


if __name__ == "__main__":
    main()
