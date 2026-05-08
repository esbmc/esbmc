class CM:
    def __enter__(self):
        return self
    def __exit__(self, et, ev, tb) -> bool:
        return True


def main() -> None:
    suppressed = False
    try:
        with CM():
            raise ValueError("x")
        suppressed = True
    except ValueError:
        pass
    assert suppressed
main()
