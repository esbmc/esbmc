class Car:
    def __init__(self, brand, speed):
        self.brand = brand
        self.speed = speed

cars = [
    Car("Toyota", 120),
    Car("Honda", 130),
    Car("Ford", 110)
]

sorted_cars = sorted(cars, key=lambda c: c.speed)

# Check if the list is sorted by speed
assert sorted_cars == sorted(cars, key=lambda c: c.speed), "List is not sorted correctly!"
