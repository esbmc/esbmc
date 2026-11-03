class A:
    pass


assert issubclass(A, int | (str, ))
