class Box:
    def __init__(self, value: bool):
        self.value = value

    def __bool__(self) -> bool:
        return self.value


x = Box(True)
y = 0

if x:
    y = 1
else:
    y = 2

assert y == 1
