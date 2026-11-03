import os


def count(top: str) -> int:
    n = 0
    for dirpath, dirnames, filenames in os.walk(top):
        n += 1
    return n


assert count(".") == 0
