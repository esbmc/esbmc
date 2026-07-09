# Python raises a catchable ZeroDivisionError on division by zero. ESBMC models
# it as a guarded exception raise (converter_binop.cpp), so a division by zero
# that is caught by try/except is SAFE — no uncaught exception escapes.
# Covers /, //, and % since all three raise ZeroDivisionError in Python.
def main() -> None:
    caught = 0

    try:
        _ = 10 / 0
    except ZeroDivisionError:
        caught += 1

    try:
        _ = 10 // 0
    except ZeroDivisionError:
        caught += 1

    try:
        _ = 10 % 0
    except ZeroDivisionError:
        caught += 1

    assert caught == 3


main()
