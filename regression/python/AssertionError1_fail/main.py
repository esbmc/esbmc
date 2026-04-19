x = 5
y = 10

try:
    assert x > y, f"Assertion failed: {x} is not greater than {y}"
except AssertionError as e:
    print("Caught an AssertionError:", e)

print("Program continues running after handling the error.")

