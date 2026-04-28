# Test __str__ returning plain self.name (no f-string)
class Item:
    def __init__(self, label: str):
        self.label = label

    def __str__(self) -> str:
        return self.label

obj = Item("hello")
assert str(obj) == "hello"
