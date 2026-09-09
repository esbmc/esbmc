# Each module gets its own Preprocessor, and only the signature tables crossed
# the boundary: the staticmethod flag did not, so the importing module stripped
# a real parameter as self again (#7546).
from lib_c import C


def main() -> None:
    d = C()
    assert d.add(10, 4) == 6
    assert C.add(10, 4) == 6


main()
