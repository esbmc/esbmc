from enum import Enum


class TrafficLight(Enum):
    RED = 1


def test_invalid_attribute():
    # Should fail verification
    assert TrafficLight.RED.invalid == 0
