# Regression for issue #5904: a missing method on a user-class instance must
# raise a catchable AttributeError.
class Foo:
    def bar(self) -> int:
        return 1


def main():
    f = Foo()
    f.baz()


main()
