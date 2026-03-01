from enum import Enum


class A(Enum):
    X = 1


class B(Enum):
    Y = 1


def test_hash_collision():
    assert hash(A.X) != hash(B.Y)
