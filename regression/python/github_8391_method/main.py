class A:
    def unsupported(self, x: int) -> int:
        match x:
            case int():
                return 1
        return 0


assert A().unsupported(1) == 1
