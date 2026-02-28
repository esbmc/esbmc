from enum import Enum


class A(Enum):
    X = 1


class B(Enum):
    X = 1


def test_cross_enum_equality():
    assert (A.X == B.X) is False
