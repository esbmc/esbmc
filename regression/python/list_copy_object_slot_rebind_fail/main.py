class Car:
    def __init__(self, speed: int):
        self.speed = speed


cars = [Car(130), Car(120)]
d = cars.copy()
d[0] = Car(5)
assert cars[0].speed == 5
