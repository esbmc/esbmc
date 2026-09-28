# github.com/esbmc/esbmc/issues/6264
# A bytes receiver has no list mutators; routing bytes.append into the list
# model used to crash. It must raise AttributeError (uncaught -> FAILED).
def main() -> None:
    b = b"abc"
    b.append(1)


main()
