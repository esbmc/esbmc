class Car:
    def __init__(self, speed: int):
        self.speed = speed


cars = [Car(120)]
assert cars[0].speed == 999
