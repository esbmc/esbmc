class Box:
    def __init__(self):
        self.value = 7


obj: int = Box()
_ = obj.value
assert True
