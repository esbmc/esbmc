import os


def count(top: str) -> int:
    n = 0
    for dirpath, dirnames, filenames in os.walk(top):
        assert dirpath == top
        n += len(dirnames) + len(filenames) + 1
    return n


assert count("nodir") >= 0
