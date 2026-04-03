z: complex = 3 + 4j

# Uncaught ZeroDivisionError: complex / real zero
result: complex = z / 0.0

assert False  # unreachable if exception is correctly raised
