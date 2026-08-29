class C:
    def __init__(self):
        self.a = 1


d = {"k": C()}
x = d["k"].no_such_attr_zzz
