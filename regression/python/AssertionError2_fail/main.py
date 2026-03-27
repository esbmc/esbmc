import random

x = random.randint(1, 1000)
y = random.randint(1, 1000)

try:
    assert x > y, f"Assertion failed: {x} is not greater than {y}"
except AssertionError as e:
    print("Caught an AssertionError:", e)

print("Program continues running after handling the error.")
