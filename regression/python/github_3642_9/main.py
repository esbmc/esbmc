from enum import Enum


class TrafficLight(Enum):
    RED = 1


def takes_enum(x: TrafficLight):
    assert x.name == "RED"


def test_struct_integrity():
    takes_enum(TrafficLight.RED)
