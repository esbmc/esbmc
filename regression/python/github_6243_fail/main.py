class Service:
    def __init__(self, name):
        self._name = name


g = None
g = Service("hi")
assert g._name == "bye"
