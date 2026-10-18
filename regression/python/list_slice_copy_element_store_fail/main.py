class Car:
    def __init__(self, speed: int):
        self.speed = speed


cars = [Car(1), Car(2)]
out = cars[:]
out[1] = out[0]
assert cars[1].speed == 1
