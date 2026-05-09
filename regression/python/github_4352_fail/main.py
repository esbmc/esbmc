class CMFalse:
    def __enter__(self):
        return self
    def __exit__(self, et, ev, tb) -> bool:
        return False


def main() -> None:
    # Negative: __exit__ returns False, so the exception still propagates.
    suppressed = False
    try:
        with CMFalse():
            raise ValueError("x")
        suppressed = True
    except ValueError:
        pass
    assert suppressed
main()
