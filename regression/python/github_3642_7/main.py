from enum import Enum


class TrafficLight(Enum):
    RED = 1


def test_invalid_attribute():
    # Never called, so its unconvertible body does not stop verification
    # (#8391).
    assert TrafficLight.RED.invalid == 0
