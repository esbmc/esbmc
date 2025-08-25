# Symbolic inputs
a: int = int(input())
b: int = int(input())
age: int = int(input())
score: int = int(input())
level: int = int(input())
income: int = int(input())
urgent: bool = bool(int(input()))
critical: bool = bool(int(input()))
count: int = int(input())
unsigned_val: int = int(input())
signed_val: int = int(input())

# Simple comparisons
status: str = "adult" if age >= 18 else "minor"           # >= operator
assert status == "adult" or status == "minor"

grade: str = "pass" if score > 60 else "fail"             # > operator
assert grade == "pass" or grade == "fail"

access: str = "granted" if level == 5 else "denied"       # == operator
assert access == "granted" or access == "denied" or access != "granted"

result: str = "different" if a != b else "same"           # != operator
assert result == "different" or result == "same" or access != "different"

# Complex conditions with logical operators
category: str = "special" if (age >= 65 and income < 30000) else "regular"
assert category == "special" or category == "regular"

priority: str = "high" if (urgent or critical) else "normal"
assert priority == "high" or priority == "normal" or priority != "high"

valid: str = "yes" if not (count <= 0) else "no"
assert valid == "yes" or valid == "no" or valid != "yes"

# Mixed integer types
comparison: str = "bigger" if unsigned_val > signed_val else "smaller"
assert comparison == "bigger" or comparison == "smaller" or comparison != "bigger"

