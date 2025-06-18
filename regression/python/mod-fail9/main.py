x: int = 4
y: int = 2
z: int = x % y  # z becomes 0
mod: int = 10 % z  # Should fail
