class Vehicle:
    # Class attributes with automatic type inference
    wheels = 4
    manufacturer = "Generic Motors"

    def __init__(self, model: str, year: int):
        self.model = model
        self.year = year

    def get_info(self) -> str:
        return f"{self.year} {self.model}"

    def get_age(self, current_year: int) -> int:
        return current_year - self.year


class Car(Vehicle):
    vehicle_type = "Car"

    def __init__(self, model: str, year: int, doors: int):
        super().__init__(model, year)
        self.doors = doors

    def get_info(self) -> str:
        base_info = super().get_info()
        return f"{base_info} with {self.doors} doors"


def test_oop_features() -> None:
    # Test class attribute access
    assert Vehicle.wheels == 4
    assert Vehicle.manufacturer == "Generic Motors"

    # Test instance creation and methods
    my_vehicle = Vehicle("Sedan", 2020)
    assert my_vehicle.get_info() == "2020 Sedan"
    assert my_vehicle.get_age(2025) == 5

    # Test inheritance and super() calls
    my_car = Car("SportsCar", 2022, 2)
    assert my_car.get_info() == "2022 SportsCar with 2 doors"
    assert Car.vehicle_type == "Car"
    assert my_car.wheels == 4  # Inherited class attribute

    # This assertion will fail - demonstrating ESBMC can verify OOP logic
    assert my_car.get_age(2025) == 4  # Should be 3, not 4


test_oop_features()
