import pytest

class Fruit:
    def __init__(self, name: str):
        self.name = name

@pytest.fixture
def my_fruit() -> Fruit:
    return Fruit("apple")

@pytest.fixture
def fruit_basket(my_fruit: Fruit) -> list[Fruit]:
    return [Fruit("banana"), my_fruit]

def test_my_fruit_in_basket(my_fruit: Fruit, fruit_basket: list[Fruit]) -> None:
    assert my_fruit in fruit_basket

f1 = Fruit("banana")
f2 = Fruit("apple")
assert f1.name == f2.name