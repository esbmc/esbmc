# github.com/esbmc/esbmc/issues/6264
# Calling a list method on a str must raise AttributeError (not crash / not
# route into the list model). Uncaught -> VERIFICATION FAILED.
def main() -> None:
    s = "abc"
    s.append("d")


main()
