# A sort key passed by the name of a lambda bound once, whose body reads only
# its parameter, is applied like the lambda itself.
class Car:
    def __init__(self, speed: int):
        self.speed = speed


cars = [Car(120), Car(130), Car(110)]
k = lambda c: c.speed
s = sorted(cars, key=k)
m = max(cars, key=k)
assert s[0].speed == 110 and m.speed == 130
