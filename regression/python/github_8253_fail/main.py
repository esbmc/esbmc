class T:
    def __init__(self) -> None:
        self.n: int = 1


t = T()
getattr(t, "total")
