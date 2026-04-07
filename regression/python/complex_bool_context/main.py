# Test complex truthiness in control flow contexts.

# Zero complex in if-condition: falsy.
value = 0
if complex(0, 0):
    value = 1
assert value == 0

# Nonzero complex in if-condition: truthy.
value2 = 0
if complex(1, 0):
    value2 = 1
assert value2 == 1

# Purely imaginary is truthy.
value3 = 0
if complex(0, 1):
    value3 = 1
assert value3 == 1

# While loop with complex condition: runs once.
counter = 0
z = complex(1, 0)
while z:
    counter = counter + 1
    z = complex(0, 0)
assert counter == 1

# Ternary with complex condition.
r1 = 10 if complex(1, 0) else 20
assert r1 == 10
r2 = 10 if complex(0, 0) else 20
assert r2 == 20

# Bool conversion: bool(complex) follows truthiness rules.
assert bool(complex(1, 0)) == True
assert bool(complex(0, 1)) == True
assert bool(complex(0, 0)) == False
assert bool(complex(float("nan"), 0)) == True
assert bool(complex(float("inf"), 0)) == True
