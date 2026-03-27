from typing import List


# Test multiple typed list instance attributes with various methods
class Container:

    def __init__(self):
        self.items: List[int] = []
        self.names: List[str] = []

    def add_item(self, item: int):
        self.items.append(item)

    def add_name(self, name: str):
        self.names.append(name)

    def remove_item(self, item: int):
        if item in self.items:
            self.items.remove(item)


c = Container()
c.add_item(1)
c.add_item(2)
c.add_item(3)
c.remove_item(2)

assert 1 in c.items
assert 2 not in c.items
assert 3 in c.items
