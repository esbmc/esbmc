# Tests that the handler correctly raises TypeError for unsupported
# complex operations.

# FloorDiv on complex
raised = False
try:
    _ = complex(1, 2) // complex(1, 0)  # type: ignore
except TypeError:
    raised = True
assert raised

# FloorDiv complex // int
raised = False
try:
    _ = complex(1, 2) // 2  # type: ignore
except TypeError:
    raised = True
assert raised

# Mod on complex
raised = False
try:
    _ = complex(1, 2) % complex(1, 0)  # type: ignore
except TypeError:
    raised = True
assert raised

# Mod complex % float
raised = False
try:
    _ = complex(1, 2) % 2.0  # type: ignore
except TypeError:
    raised = True
assert raised

# Ordering: < on complex
raised = False
try:
    _ = complex(1, 2) < complex(3, 4)  # type: ignore
except TypeError:
    raised = True
assert raised

# Ordering: > on complex
raised = False
try:
    _ = complex(1, 2) > complex(3, 4)  # type: ignore
except TypeError:
    raised = True
assert raised

# Ordering: <= on complex
raised = False
try:
    _ = complex(1, 2) <= complex(3, 4)  # type: ignore
except TypeError:
    raised = True
assert raised

# Ordering: >= on complex
raised = False
try:
    _ = complex(1, 2) >= complex(3, 4)  # type: ignore
except TypeError:
    raised = True
assert raised

# Ordering: complex < int
raised = False
try:
    _ = complex(1, 2) < 3  # type: ignore
except TypeError:
    raised = True
assert raised

# Pow with string exponent
raised = False
try:
    _ = complex(1, 2) ** "2"  # type: ignore
except TypeError:
    raised = True
assert raised

# Add with string
raised = False
try:
    _ = complex(1, 2) + "hello"  # type: ignore
except TypeError:
    raised = True
assert raised

# Mul with string
raised = False
try:
    _ = complex(1, 2) * "hello"  # type: ignore
except TypeError:
    raised = True
assert raised
