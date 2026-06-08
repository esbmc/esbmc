from typing import Dict, Any
import stubs.ros_mock as rospy
from stubs.ros_mock import Twist


class EmptyQueue(Exception):
    pass


class SimpleQueue:
    def __init__(self):
        self._items = []


class SimulationState:

    def __init__(self):
        self.position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.orientation = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.sensor_data = {}

class ROSComponent:
    """Base class for ROS components"""

    def __init__(self, name: str):
        self.name = name
        self.publishers: Dict[str, rospy.Publisher] = {}
        self.message_queue = SimpleQueue()

    def create_publisher(self, topic: str, msg_type, queue_size: int=10):
        self.publishers[topic] = rospy.Publisher(topic, msg_type, queue_size=queue_size)

class FPrimeComponent:
    """Base class for F Prime components"""

    def __init__(self, name: str):
        self.name = name

class ROSFPrimeBridge:
    """Minimal bridge stub for Shedskin."""

    def __init__(self):
        self.running = False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

class RoverSimulation:
    """Main simulation class integrating ROS and F Prime components"""

    def __init__(self):
        rospy.init_node('rover_simulation')
        self.state = SimulationState()
        self.bridge = ROSFPrimeBridge()
        self.ros_components: Dict[str, ROSComponent] = {}
        self.fprime_components: Dict[str, FPrimeComponent] = {}

    def add_ros_component(self, component: ROSComponent):
        self.ros_components[component.name] = component

    def add_fprime_component(self, component: FPrimeComponent):
        self.fprime_components[component.name] = component

    def start(self):
        """Start the simulation"""
        self.bridge.start()
        rospy.loginfo('Simulation started')

    def stop(self):
        """Stop the simulation"""
        self.bridge.stop()
        rospy.loginfo('Simulation stopped')

    def update_state(self, new_state: dict):
        """Update simulation state"""
        if 'position' in new_state:
            self.state.position = new_state['position']
        if 'orientation' in new_state:
            self.state.orientation = new_state['orientation']
        if 'sensor_data' in new_state:
            self.state.sensor_data = new_state['sensor_data']

class MotorController(ROSComponent):
    """Example ROS component for motor control"""

    def __init__(self):
        ROSComponent.__init__(self, 'motor_controller')
        self.create_publisher('/cmd_vel', Twist)

    def set_velocity(self, linear: float, angular: float):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.publishers['/cmd_vel'].publish(msg)

class NavigationComponent(FPrimeComponent):
    """Example F Prime component for navigation"""

    def __init__(self):
        FPrimeComponent.__init__(self, 'navigation')

def main():
    sim = RoverSimulation()
    motor_controller = MotorController()
    nav_component = NavigationComponent()
    sim.add_ros_component(motor_controller)
    sim.add_fprime_component(nav_component)

    sim.start()
    steps = 3
    i = 0
    while i < steps:
        sim.update_state({'position': {'x': 1.0, 'y': 2.0, 'z': 0.0}})
        motor_controller.set_velocity(0.5, 0.1)
        i = i + 1
    sim.stop()
if __name__ == '__main__':
    main()
