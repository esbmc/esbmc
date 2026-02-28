from enum import Enum


class TrafficLight(Enum):
    RED = 1
    YELLOW = 2
    GREEN = 3


def can_go(light: TrafficLight) -> bool:
    if light == TrafficLight.GREEN:
        return True
    elif light == TrafficLight.RED:
        return False
    elif light == TrafficLight.YELLOW:
        return False
    else:
        assert False, "Unknown traffic light state"


def test_traffic_light():
    assert can_go(TrafficLight.GREEN) is True
    assert can_go(TrafficLight.RED) is False
    assert can_go(TrafficLight.YELLOW) is False

    assert TrafficLight.RED.value == 1
    assert TrafficLight.YELLOW.value == 2
    assert TrafficLight.GREEN.value == 3

    assert TrafficLight.RED.name == "RED"
    assert TrafficLight.GREEN.name == "GREEN"


if __name__ == "__main__":
    test_traffic_light()
