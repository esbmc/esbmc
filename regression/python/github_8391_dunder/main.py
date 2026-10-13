class A:
    def __eq__(self, other: object) -> bool:
        match other:
            case A():
                return True
        return False


assert A() == A()
