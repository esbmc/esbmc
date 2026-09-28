import os


def abspath(filename):
    return os.path.abspath(filename)


assert abspath("x") is not None
