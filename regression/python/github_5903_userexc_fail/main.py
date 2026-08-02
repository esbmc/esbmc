# Regression for issue #5903: a user-raised exception escaping main must be
# named in the verdict ("uncaught exception: MyError").
class MyError(Exception):
    pass


def main():
    raise MyError()


main()
