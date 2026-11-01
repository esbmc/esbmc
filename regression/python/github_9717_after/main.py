class A:
    def m(self):
        return 1


class B(A):
    pass


def add(to):
    to.append(B())


lst = [A()]
assert lst[0].m() == 1
add(lst)
assert len(lst) == 2
