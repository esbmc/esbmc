# github #7745: the negative twin. Typing the parameter from the list literal
# must not make the attribute's value provable -- only its type is known here.
class Car:
    def __init__(self, speed: int):
        self.speed = speed


f = lambda c: c.speed
a = Car(120)
cars = [a]
x = f(cars[0])
assert x == 999
