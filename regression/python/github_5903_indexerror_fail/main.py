# Regression for issue #5903: an uncaught IndexError escaping main must produce
# a type-named verdict ("uncaught exception: IndexError"), distinct from other
# exception families.
def main():
    a = [1, 2, 3]
    z = a[5]


main()
