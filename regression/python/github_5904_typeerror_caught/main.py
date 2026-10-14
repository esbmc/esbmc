# Regression for issue #5904: a TypeError from `int + str` must be catchable by
# a matching `except TypeError` handler.
def main():
    try:
        y = 1 + "s"
    except TypeError:
        pass


main()
