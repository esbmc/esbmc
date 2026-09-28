# A lambda returning an int kept a double signature, so a value above 2**53
# was rounded on the way in and on the way out (#7745).
class Car:
    def __init__(self, speed: int):
        self.speed = speed


f = lambda c: c.speed
g = lambda n: n + 0
a = Car(9007199254740993)
cars = [a]
x: int = 9007199254740993
assert f(cars[0]) != 9007199254740992
assert g(x) != 9007199254740992
assert g(9007199254740993) != 9007199254740992
