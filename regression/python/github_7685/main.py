# A constructor call in element position arrives as a value struct, but the
# element read emits `*(Cls **)item->value`, so the field came back as a
# pointer reinterpretation of its own value.
class Car:
    def __init__(self, speed: int):
        self.speed = speed


cars = [Car(120), Car(130)]
assert cars[0].speed == 120
assert cars[1].speed == 130
