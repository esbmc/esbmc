# github #7745: rebinding a name to the same thing is ordinary Python. The
# list and the element are each bound twice; the bindings agree, so the type is
# still determined.
class Car:
    def __init__(self, speed: int):
        self.speed = speed


f = lambda c: c.speed
a = Car(120)
cars = [a]
cars = [a]
x = f(cars[0])
assert x == 120
