def safe_divide(a: int, b: int) -> int:
    try:
        result = a // b
        return result
    except ZeroDivisionError as e:
        return -1

def test_exception_handling() -> None:
    # Normal case
    assert safe_divide(10, 2) == 5

    # Division by zero case
    assert safe_divide(10, 0) == -1

    # This assertion will fail - demonstrating ESBMC can verify exception paths
    assert safe_divide(8, 0) == 0  # Should be -1, not 0

test_exception_handling()
