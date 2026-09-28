# github #7745: a lambda applied to a list element whose type the element-type
# registry has not recorded yet. The registry is filled in conversion order, so
# a lambda bound before the list it is called with saw nothing there and its
# parameter stayed double, which the solver then met as a Car pointer.
class Car:
    def __init__(self, speed: int):
        self.speed = speed


f = lambda c: c.speed
a = Car(120)
cars = [a]
x = f(cars[0])
assert x == 120
