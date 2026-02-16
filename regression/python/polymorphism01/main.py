from typing import Union

class Vehicle:
    def move(self) -> str:
        # This should never be called if only subclasses are instantiated
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

# Test cases - directly test each vehicle type
car: Car = Car()
result_car: str = car.move()
assert result_car == "Driving on the road", f"Expected 'Driving on the road', got '{result_car}'"

boat: Boat = Boat()
result_boat: str = boat.move()
assert result_boat == "Sailing on the water", f"Expected 'Sailing on the water', got '{result_boat}'"

airplane: Airplane = Airplane()
result_airplane: str = airplane.move()
assert result_airplane == "Flying in the sky", f"Expected 'Flying in the sky', got '{result_airplane}'"
