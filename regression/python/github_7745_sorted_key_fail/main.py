# A key lambda reading an attribute of a list element was bound to a name
# whose parameter defaulted to double, and the solver rejected the object
# pointer passed to it (#7745).
class Car:

    def __init__(self, speed: int, name: str):
        self.speed = speed
        self.name = name


cars = [Car(3, "b"), Car(1, "c"), Car(2, "a")]
d = sorted(cars, key=lambda c: -c.speed)
assert d[0].speed == 1
