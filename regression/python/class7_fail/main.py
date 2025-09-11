class Car:

    def __init__(self, id_number: int, mileage: int):
        assert mileage >= 0, "mileage cannot be negative"
        self.id_number = id_number
        self.mileage = mileage

    def start(self) -> str:
        return "Engine started"

    def drive(self, distance: int) -> int:
        assert distance > 0, "distance must be positive"
        self.mileage += distance
        return self.mileage

    def get_mileage(self) -> int:
        return self.mileage

    def needs_service(self) -> bool:
        return self.mileage >= 10000


# Create instances
car1 = Car(101, 9500)
car2 = Car(202, 10000)
car3 = Car(303, 0)

# Method output tests
assert car1.start() == "Engine started"
assert car2.start() == "Engine started"

assert car1.drive(400) == 9900
assert car1.get_mileage() == 9900
assert not car1.needs_service()

assert car2.drive(200) == 10200
assert car2.get_mileage() == 10200
assert car2.needs_service()

assert car3.drive(1000) == 1000
assert car3.get_mileage() == 1001  # should fail
assert not car3.needs_service()
