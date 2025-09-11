class Vehicle:

    def move(self):
        assert False


class Car(Vehicle):

    def move(self):
        return "Driving on the road"


class Boat(Vehicle):

    def move(self):
        return "Sailing on the water"


class Airplane(Vehicle):

    def move(self):
        return "Flying in the sky"


def test_vehicle_movement(vehicle, expected_movement):
    result = vehicle.move()
    assert result == expected_movement


# Test cases
car = Car()
boat = Boat()
airplane = Airplane()

test_vehicle_movement(car, "Driving on the road1")
test_vehicle_movement(boat, "Sailing on the water")
test_vehicle_movement(airplane, "Flying in the sky")
