# extend() applies one copy length to every element, so the frontend supplies a
# width only when all recorded elements share it. Each arm is exercised here.

# Uniform scalar width: the frontend passes 8.
ints = [1, 2]
ints.extend([3, 4])
assert ints == [1, 2, 3, 4]

floats = [1.5]
floats.extend([2.5, 3.5])
assert floats == [1.5, 2.5, 3.5]

# Mixed widths: no single length, so the model keeps its elem->size fallback.
mixed = [1]
mixed.extend([2, "abc"])
assert mixed[1] == 2
assert mixed[2] == "abc"

# Non-scalar elements: same fallback.
nested = [[1, 2]]
nested.extend([[3, 4]])
assert nested[1][0] == 3

# Strings differ in width element to element.
words = ["a"]
words.extend(["bb", "ccc"])
assert words == ["a", "bb", "ccc"]
