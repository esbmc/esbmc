class Car:
    def __init__(self, speed: int):
        self.speed = speed


cars = [Car(1), Car(2)]
out = cars[:]
out[1] = out[0]
assert out[1].speed == 1
assert cars[1].speed == 2

rows = [[1], [2]]
copy = rows[:]
copy[1] = copy[0]
assert copy[1] == [1]
assert rows[1] == [2]

fleet = [Car(130), Car(120), Car(125)]
by_speed = sorted(fleet, key=lambda c: c.speed)
assert by_speed[0].speed == 120
assert by_speed[2].speed == 130
assert fleet[1].speed == 120
