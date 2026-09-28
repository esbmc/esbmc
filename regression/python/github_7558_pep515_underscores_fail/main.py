# A trailing underscore is not a separator, so int() must still raise.
def main() -> None:
    x = int("100_")
    assert x == 100


main()
