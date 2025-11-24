def parse_number(x: int) -> int:
    if x < 0:
        raise ValueError("Negative number!")
    return x

result = 0
try:
    result = parse_number(-1)
except BaseException as e:
    print("Caught by BaseException:", e)

assert result == 0
