# The reflected A.__le__ decides both comparisons (#10156).
class A:
    def __le__(self, o: int) -> bool:
        return o >= 5


a = A()
assert 7 >= a
assert 2 >= a
