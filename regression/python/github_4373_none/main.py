class CMNone:
    def __enter__(self):
        return self
    def __exit__(self, et, ev, tb):
        pass  # implicit None return -> exception must propagate


def main() -> None:
    caught = False
    try:
        with CMNone():
            raise ValueError("x")
    except ValueError:
        caught = True
    assert caught
main()
