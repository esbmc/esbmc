from enum import Enum


class TrafficLight(Enum):
    RED = 1


def test_nested_access():
    name = TrafficLight.RED.name
    assert name == "RED"
