# `result` is a float on the fall-through path and a str when the one-armed if
# is taken, and `.upper()` is then called on it. Typing it by the last
# assignment alone made the float path's AttributeError unreachable and the
# program verified. A divergence the tagged path cannot carry is refused
# instead (#8263).
def main():
    a = 1.0
    b = 2.0
    result = a + b
    if nondet_bool():
        a = "Hello"
        b = "World"
        result = a + b
    shout = result.upper()


main()
