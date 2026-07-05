# Reusing a list-variable name across comprehensions whose sources hold
# different classes must not hang the frontend. The module-wide element-class
# fixpoint previously spun forever once such a name resolved to a conflict
# (#4805 review finding).


class A:
    def __init__(self):
        self.x = 1


class B:
    def __init__(self):
        self.x = 2


def run():
    s1 = [A(), A()]
    s2 = [B(), B()]
    v = [p for p in s1]
    v = [q for q in s2]
    return len(v)


assert run() == 2
