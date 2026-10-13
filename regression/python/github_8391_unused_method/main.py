class A:
    def unsupported(self, x: int) -> int:
        match x:
            case 1:
                return 1
        return 0

    def used(self) -> int:
        return 1


assert A().used() == 1
