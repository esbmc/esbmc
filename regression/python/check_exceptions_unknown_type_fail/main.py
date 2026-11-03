# BufferError is not modelled, so ESBMC cannot tell which raised types are
# one. Ignoring the name would hide them, so it is rejected.
def main() -> None:
    l = [1, 2]
    x = l[2]


main()
