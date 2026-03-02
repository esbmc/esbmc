def test_value(x: int) -> int:
    if x < 0:
        raise ValueError("Negative")
    return x


try:
    test_value(-1)
except ValueError as e:
    print("Caught by ValueError:", e)
except Exception as e:
    print("Caught by Exception:", e)
