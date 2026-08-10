# A closure that escapes its defining scope: make_adder returns add, which
# captures n. Calling it must yield 8. ESBMC currently leaves the call
# unconstrained -- n is not bound -- so both `== 8` and `== 3` are refutable
# (#6256). Same root cause as #6640: function values are static aliases, so
# there is no environment to carry n out of make_adder.


def make_adder(n):
    def add(x):
        return x + n

    return add


add5 = make_adder(5)
assert add5(3) == 8
