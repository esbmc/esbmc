# Regression for issue #5903: an uncaught ZeroDivisionError escaping main must
# produce a type-named verdict ("uncaught exception: ZeroDivisionError").
def main():
    raise ZeroDivisionError("boom")


main()
