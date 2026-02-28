class Animal:

    def __init__(self, name: str):
        self.name = name
        self.energy = 100

    def move(self) -> int:
        self.energy -= 10
        return self.energy


class Dog(Animal):

    def __init__(self, name: str, breed: str):
        super().__init__(name)
        self.breed = breed

    def bark(self) -> str:
        self.energy -= 5
        return "Woof!"


def test_inheritance() -> None:
    dog = Dog("Buddy", "Golden Retriever")

    # Test inherited method
    assert dog.energy == 100
    energy_after_move: int = dog.move()
    assert energy_after_move == 90

    # Test derived class method
    sound: int = dog.bark()
    assert dog.energy == 85
    assert sound == "Woof!"


test_inheritance()
