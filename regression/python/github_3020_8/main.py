def process(value: int) -> int:
    return value * 2

result1 = process(5)  # Correct
result2 = process("wrong")  # Should fail here
