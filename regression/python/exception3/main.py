def process_data(x: int) -> int:
    if x < 0:
        raise ValueError("Negative value")
    elif x == 0:
        raise ZeroDivisionError("Zero value")
    return x

try:
    process_data(-1)
except Exception as e:
    print("Caught:", e)
