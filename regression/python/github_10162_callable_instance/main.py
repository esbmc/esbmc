class Adder:
    def __call__(self, a, b):
        return a + b


max = Adder()
assert max(2, 1) == 3
