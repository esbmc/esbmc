class A:
    pass


def make() -> A:
    return A()


a = make()
print(type(a))
