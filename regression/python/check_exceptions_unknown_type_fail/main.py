# LookupError is not modelled, so ESBMC cannot tell that IndexError is one.
# Ignoring the name would hide the IndexError, so it is rejected.
def main() -> None:
    l = [1, 2]
    x = l[2]


main()
