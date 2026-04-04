z = complex(1, 2)
z += 1.0
assert z == complex(3.0, 2.0)  # wrong: real should be 2.0
