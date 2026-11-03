def outer():
    def a():
        return b() + 1

    def b():
        return 1

    return a()


assert outer() == 3
