class A:
    def m(self):
        return 1


class B(A):
    pass


def add(to):
    to.append(B())


lst = [A()]


def first():
    return lst[0].m()


add(lst)
assert first() == 1
