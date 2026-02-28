from enum import Enum


class TrafficLight(Enum):
    RED = 1


def test_enum_vs_int():
    assert (TrafficLight.RED == 1) is False
    assert (1 == TrafficLight.RED) is False
