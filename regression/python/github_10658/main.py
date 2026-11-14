class C:
    def __init__(self):
        self.b = "xy"


o = C()
assert o.b == "xy"
o.b = 10
assert o.b == 10
