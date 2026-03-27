# Global used in method body (not just returned)
counter: int = 10


class Incrementer:

    def get_base(self) -> int:
        return counter

    def compute(self, delta: int) -> int:
        return counter + delta


obj = Incrementer()
assert obj.get_base() == 10
assert obj.compute(5) == 15
