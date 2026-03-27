# Test case 11: Nested attribute type resolution with Inheritance

class BaseComponent:
    def get_base_id(self) -> int:
        return 1

class Component(BaseComponent):
    def __init__(self) -> None:
        pass

class System:
    def __init__(self) -> None:
        self.comp: Component = Component()

    def check(self) -> int:
        return self.comp.get_base_id()

s = System()
assert s.check() == 1