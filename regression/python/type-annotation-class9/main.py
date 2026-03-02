class A:

    def __init__(self, s):
        self.s = s


a = A("hello")
assert a.s == "hello"
