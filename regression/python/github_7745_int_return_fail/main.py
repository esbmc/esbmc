# A lambda returning an int kept a double signature, so a value above 2**53
# was rounded on the way in and on the way out (#7745).
class Car:
    def __init__(self, speed: int):
        self.speed = speed


f = lambda c: c.speed
a = Car(9007199254740993)
cars = [a]
assert f(cars[0]) == 9007199254740992
