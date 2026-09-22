# github #7745: the element is a constructor call directly, with no name to
# hop through -- the base case of the AST fallback.
class Car:
    def __init__(self, speed: int):
        self.speed = speed


f = lambda c: c.speed
cars = [Car(120)]
x = f(cars[0])
assert x == 120
