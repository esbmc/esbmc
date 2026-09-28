# github #7745: a bare annotation is an AnnAssign carrying a JSON null. It
# binds nothing, and reading it as an object threw.
class Car:
    def __init__(self, speed: int):
        self.speed = speed


f = lambda c: c.speed
a: Car
a = Car(120)
cars = [a]
x = f(cars[0])
assert x == 120
