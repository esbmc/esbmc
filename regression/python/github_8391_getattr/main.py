class A:
    def unsupported(self, x: int) -> int:
        match x:
            case 1:
                return 1
        return 0


assert getattr(A(), "unsupported")(1) == 1
