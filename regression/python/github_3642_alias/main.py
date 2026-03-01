from enum import Enum as E


class A(E):
    X = 1


def test_alias_base():
    assert A.X.value == 1


if __name__ == "__main__":
    test_alias_base()
