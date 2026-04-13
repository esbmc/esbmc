# example_14_concurrency.py
import random
import threading
import time
from queue import Queue
from typing import List, Callable
from dataclasses import dataclass

# 1. Guarded Commands Pattern
class Action:
    def pre(self) -> bool:
        """Return True if the action is enabled."""
        pass

    def act(self) -> None:
        """Execute the action."""
        pass

class Send(Action):
    def __init__(self, queue: List[int]):
        self.queue = queue

    def pre(self) -> bool:
        return True

    def act(self) -> None:
        number = random.randrange(0, 151)
        print(f'Sending: {number}')
        self.queue.append(number)

class Receive(Action):
    def __init__(self, queue: List[int]):
        self.queue = queue

    def pre(self) -> bool:
        return len(self.queue) > 0

    def act(self) -> None:
        number = self.queue.pop(0)
        print(f'Receiving: {number}')
        assert number < 100

def run_guarded_commands():
    queue: List[int] = []
    processes = [Send(queue), Receive(queue)]

    for _ in range(5):  # Run 5 iterations for demonstration
        enabled = [p for p in processes if p.pre()]
        if enabled:
            choice = random.choice(enabled)
            choice.act()
        time.sleep(0.5)  # Small delay for demonstration

# 2. Threading Pattern
class Sender(threading.Thread):
    def __init__(self, queue: Queue):
        super().__init__()
        self.queue = queue
        self.running = True

    def run(self):
        while self.running:
            number = random.randint(0, 150)
            print(f"Sending: {number}")
            self.queue.put(number)
            time.sleep(1)  # Simulate work

    def stop(self):
        self.running = False

class Receiver(threading.Thread):
    def __init__(self, queue: Queue):
        super().__init__()
        self.queue = queue
        self.running = True

    def run(self):
        while self.running:
            number = self.queue.get()
            print(f"Receiving: {number}")
            assert number < 100
            self.queue.task_done()

    def stop(self):
        self.running = False

# 3. Publish/Subscribe Pattern
@dataclass
class Message:
    content: str
    priority: int

class Topic:
    def __init__(self, name: str):
        self.name = name
        self.subscribers: List[Callable[[Message], None]] = []

    def subscribe(self, callback: Callable[[Message], None]):
        self.subscribers.append(callback)

    def publish(self, message: Message):
        print(f"Publishing to {self.name}: {message}")
        for subscriber in self.subscribers:
            subscriber(message)

# 4. Clock-Driven Pattern
class ClockDrivenSystem:
    def __init__(self):
        self.tick = 0
        self.running = True

    def process1(self):
        print(f"Process 1 at tick {self.tick}")

    def process2(self):
        print(f"Process 2 at tick {self.tick}")

    def process3(self):
        print(f"Process 3 at tick {self.tick}")

    def run(self, ticks: int):
        while self.tick < ticks:
            print(f"\nTick {self.tick}")
            self.process1()
            self.process2()
            self.process3()
            self.tick += 1
            time.sleep(0.5)  # Simulate clock cycle

def demonstrate_concurrency():
    print("1. Demonstrating Guarded Commands Pattern:")
    run_guarded_commands()

    print("\n2. Demonstrating Threading Pattern:")
    queue = Queue()
    sender = Sender(queue)
    receiver = Receiver(queue)

    sender.start()
    receiver.start()

    # Let it run for a few seconds
    time.sleep(3)

    sender.stop()
    receiver.stop()
    sender.join()
    receiver.join()

    print("\n3. Demonstrating Publish/Subscribe Pattern:")
    topic = Topic("updates")

    def subscriber1(msg: Message):
        print(f"Subscriber 1 received: {msg}")

    def subscriber2(msg: Message):
        print(f"Subscriber 2 received: {msg}")

    topic.subscribe(subscriber1)
    topic.subscribe(subscriber2)

    topic.publish(Message("Hello", 1))
    topic.publish(Message("World", 2))

    print("\n4. Demonstrating Clock-Driven Pattern:")
    system = ClockDrivenSystem()
    system.run(3)  # Run for 3 ticks

if __name__ == "__main__":
    demonstrate_concurrency()
