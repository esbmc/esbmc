class LoopClass:

    def __init__(self):
        pass

    def looper(self) -> int:
        x: int = 0
        for i in range(5):
            x = i
        return x
