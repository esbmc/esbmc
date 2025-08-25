# Input variables with type hints and initialization
a: int = 3
b: int = 5
age: int = 20
score: int = 75
level: int = 5
income: int = 25000
urgent: bool = False
critical: bool = True
count: int = 10
unsigned_val: int = 7   # assuming non-negative
signed_val: int = -3

# Simple comparisons
status: str = "adult" if age >= 18 else "minor"           # >= operator
assert status == "adult" or status == "minor"

grade: str = "pass" if score > 60 else "fail"             # > operator
assert grade == "pass" or grade == "fail"

access: str = "granted" if level == 5 else "denied"       # == operator
assert access == "granted" or access == "denied"

result: str = "different" if a != b else "same"           # != operator
assert result == "different" or result == "same"

# Complex conditions with logical operators
category: str = "special" if (age >= 65 and income < 30000) else "regular"
assert category == "special" or category == "regular"

priority: str = "high" if (urgent or critical) else "normal"
assert priority == "high" or priority == "normal"

valid: str = "yes" if not (count <= 0) else "no"
assert valid == "yes" or valid == "no"

# Mixed integer types
comparison: str = "bigger" if unsigned_val > signed_val else "smaller"
assert comparison == "bigger" or comparison == "smaller"

print("All checks passed!")

