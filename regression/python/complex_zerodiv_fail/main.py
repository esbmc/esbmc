z: complex = 1 + 2j
w: complex = 0 + 0j

# Uncaught ZeroDivisionError: complex / complex zero
result: complex = z / w

assert False  # unreachable if exception is correctly raised
