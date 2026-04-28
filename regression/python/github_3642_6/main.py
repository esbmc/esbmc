from enum import Enum


class TrafficLight(Enum):
    RED = 1


def test_name_string_compare():
    assert TrafficLight.RED.name == "RED"
    assert TrafficLight.RED.name != "GREEN"
