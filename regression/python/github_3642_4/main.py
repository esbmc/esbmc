from enum import Enum


class TrafficLight(Enum):
    RED = 1
    GREEN = 2


def test_ordering_not_allowed():
    assert not (TrafficLight.RED < TrafficLight.GREEN)
