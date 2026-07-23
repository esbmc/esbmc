class Service:

    def __init__(self, name):
        self._name = name


x = Service("first")
x = None
x = Service("second")
assert x._name == "second"
