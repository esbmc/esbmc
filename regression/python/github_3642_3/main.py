from enum import Enum


class TrafficLight(Enum):
    RED = 1


class Dummy:
    pass


def test_enum_vs_object():
    d = Dummy()
    assert (TrafficLight.RED == d) is False
