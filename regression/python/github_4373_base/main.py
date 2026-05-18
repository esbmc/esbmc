class Base:
    def __enter__(self):
        return self
    def __exit__(self, et, ev, tb) -> bool:
        return True


class Child(Base):
    pass


def main() -> None:
    suppressed = False
    try:
        with Child():
            raise ValueError("x")
        suppressed = True
    except ValueError:
        pass
    assert suppressed
main()
