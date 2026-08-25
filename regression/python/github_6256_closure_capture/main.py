# A closure that escapes its defining scope: make_adder returns add, which
# captures n. Calling it must yield 8 (#6256).


def make_adder(n):
    def add(x):
        return x + n

    return add


add5 = make_adder(5)
assert add5(3) == 8
