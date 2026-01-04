class Positive:
    def __init__(self, value: int):
        assert value >= 0
        self.value = value

Positive(-2)
