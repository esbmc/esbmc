# Regression for issue #5903: an uncaught KeyError escaping main must produce a
# type-named verdict ("uncaught exception: KeyError"), distinct from IndexError.
def main():
    d = {"a": 1}
    y = d["b"]


main()
