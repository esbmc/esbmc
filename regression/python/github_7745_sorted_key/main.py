# A key lambda reading an attribute of a list element was bound to a name
# whose parameter defaulted to double, and the solver rejected the object
# pointer passed to it (#7745).
class Car:

    def __init__(self, speed: int, name: str):
        self.speed = speed
        self.name = name


cars = [Car(3, "b"), Car(1, "c"), Car(2, "a")]
s = sorted(cars, key=lambda c: c.speed)
assert s[0].speed == 1 and s[1].speed == 2 and s[2].speed == 3
d = sorted(cars, key=lambda c: -c.speed)
assert d[0].speed == 3 and d[2].speed == 1
assert max(cars, key=lambda c: c.speed).speed == 3
assert min(cars, key=lambda c: c.name).speed == 2
