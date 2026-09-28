import numpy as np

# Both 1-D operands are float variables (not literals, not int). Exercises
# the dot_double backend path with variable-sourced operands.
a = np.array([1.5, 2.5, 3.5])
b = np.array([1.0, 1.0, 1.0])

result = np.dot(a, b)

assert result == 7.5
