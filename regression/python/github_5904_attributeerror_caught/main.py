# Regression for issue #5904: an AttributeError from a missing method must be
# catchable by a matching `except AttributeError` handler.
def main():
    x = 5
    try:
        x.foo()
    except AttributeError:
        pass


main()
