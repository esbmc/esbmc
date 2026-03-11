from enum import Enum


class TrafficLight(Enum):
    RED = 1


def test_symmetry():
    assert (TrafficLight.RED == TrafficLight.RED)
    assert not (TrafficLight.RED != TrafficLight.RED)
