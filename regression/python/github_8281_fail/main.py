# check_type raises TypeError when the value does not have the hinted type,
# and the violated property names the family so a caller can map it.
import _sv_verifier


def main():
    _sv_verifier.check_type("s", int)


main()
