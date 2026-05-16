class CMNested:
    """__exit__ returns None implicitly. The inner helper returning True
    must not fool the truthy-Return scan into wrapping this class for
    dynamic dispatch (which would crash on the void result)."""
    def __enter__(self):
        return self
    def __exit__(self, et, ev, tb):
        def helper():
            return True
        helper()
        return None


def main() -> None:
    caught = False
    try:
        with CMNested():
            raise ValueError("x")
    except ValueError:
        caught = True
    assert caught
main()
