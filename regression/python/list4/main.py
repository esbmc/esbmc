class Car:
    def __init__(self, brand, speed):
        self.brand = brand
        self.speed = speed

cars = [
    Car("Toyota", 120),
    Car("Honda", 130),
    Car("Ford", 110)
]

# Ensure all cars have a speed greater than 100
assert all(car.speed > 100 for car in cars), "Some cars are too slow!"
