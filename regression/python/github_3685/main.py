class Vehicle:
    def __init__(self, year: int):
        self.year = year

    def get_age(self, current_year: int) -> int:
        return current_year - self.year

class Car(Vehicle):
    def __init__(self, year: int):
        super().__init__(year)

def test_bug() -> None:
    my_car = Car(2022)
    assert my_car.get_age(2025) == 3

test_bug()
