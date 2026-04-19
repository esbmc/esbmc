from enum import Enum


class TrafficLight(Enum):
    RED = 1


def test_dict_usage():
    d = {TrafficLight.RED: 10}
    assert d[TrafficLight.RED] == 10
