from typing import Union

class Vehicle:
    def move(self) -> str:
        # Abstract method should raise an error or be marked abstract
        raise NotImplementedError("Subclasses must implement this method")

class Car(Vehicle):
    def move(self) -> str:
        return "Driving on the road"

class Boat(Vehicle):
    def move(self) -> str:
        return "Sailing on the water"

class Airplane(Vehicle):
    def move(self) -> str:
        return "Flying in the sky"

def test_vehicle_movement(vehicle: Vehicle, expected_movement: str) -> None:
    result: str = vehicle.move()
    assert result == expected_movement, f"Expected '{expected_movement}', got '{result}'"

# Test cases
car: Car = Car()
boat: Boat = Boat()
airplane: Airplane = Airplane()

test_vehicle_movement(car, "Driving on the road")
test_vehicle_movement(boat, "Sailing on the water")
test_vehicle_movement(airplane, "Flying in the sky")

