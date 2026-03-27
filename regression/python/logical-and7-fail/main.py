x: int = 2147483647  # Max int (assuming 32-bit)
y: int = -2147483648  # Min int
z: int = 0
if (x > 0 and y < 0 and z == 0):
    result: float = 1 / 0
