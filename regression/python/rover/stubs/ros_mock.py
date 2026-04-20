"""Minimal ROS stubs so Shedskin can convert ROS-dependent examples."""

class ROSInterruptException(Exception):
    pass

_shutdown = False


def init_node(name):
    return None


def loginfo(msg):
    print(msg)


class Publisher:
    def __init__(self, topic, msg_type, queue_size=10):
        self.topic = topic

    def publish(self, msg):
        return None


class Subscriber:
    def __init__(self, topic, msg_type, callback):
        self.topic = topic


class TwistLinear:
    def __init__(self):
        self.x = 0.0


class TwistAngular:
    def __init__(self):
        self.z = 0.0


class Twist:
    def __init__(self):
        self.linear = TwistLinear()
        self.angular = TwistAngular()
