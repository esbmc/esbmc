# Input variables with type hints
a:int
b:int
age: int
score: int
level: int
income: int
urgent: bool
critical: bool
count: int
unsigned_val: int  # assuming non-negative, but still an int in Python
signed_val: int

# Simple comparisons
status: str = "adult" if age >= 18 else "minor"           # >= operator
assert status == "adult" or status == "minor"
grade: str = "pass" if score > 60 else "fail"             # > operator  
assert grade == "pass" or grade == "fail"
access: str = "granted" if level == 5 else "denied"       # == operator
assert access == "granted" or "denied"
result: str = "different" if a != b else "same"           # != operator
assert result == "different" # this assertion should fail

# Complex conditions with logical operators
category: str = "special" if (age >= 65 and income < 30000) else "regular"
assert category == "special" or "regular"
priority: str = "high" if (urgent or critical) else "normal" 
assert priority == "high" or "normal"
valid: str = "yes" if not (count <= 0) else "no"
assert valid == "yes" or "no"

# Mixed integer types
comparison: str = "bigger" if unsigned_val > signed_val else "smaller"
assert comparison == "bigger" or "smaller"
