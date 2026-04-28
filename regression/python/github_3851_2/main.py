# Multiple class methods referencing globals declared after the class
class Checker:
    def check_positive(self) -> bool:
        return threshold > 0

    def check_max(self) -> bool:
        return threshold <= max_val


threshold: int = 5
max_val: int = 100

c = Checker()
assert c.check_positive()
assert c.check_max()
