class Car:
    def __init__(self, speed: int):
        self.speed = speed


cars = [Car(130), Car(120)]
d = cars[:]
d[1].speed = 7
assert cars[1].speed == 7
d[0] = Car(5)
assert d[0].speed == 5
assert cars[0].speed == 130

s = sorted(cars, key=lambda c: c.speed)
assert s[0].speed == 7
assert cars[0].speed == 130
