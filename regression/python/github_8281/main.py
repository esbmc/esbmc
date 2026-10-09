# Importing _sv_verifier used to fail outright once it defined check_type,
# because converting _matches_type needs typing.get_origin, types.UnionType
# and isinstance over a variable class (#8281).
import _sv_verifier


def main():
    _sv_verifier.check_type(1, int)
    _sv_verifier.check_type("s", str)


main()
