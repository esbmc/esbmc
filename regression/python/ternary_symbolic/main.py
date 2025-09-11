# Symbolic inputs for comprehensive verification
a: int = int(input())
b: int = int(input())
age: int = int(input())
score: int = int(input())
level: int = int(input())
income: int = int(input())
temperature: float = float(input())
weight: float = float(input())
height: float = float(input())
urgent: bool = bool(int(input()))
critical: bool = bool(int(input()))

# Test ternary with symbolic values - basic comparisons
adult_status: int = 1 if age >= 18 else 0
assert adult_status == 1 or adult_status == 0

grade_status: int = 1 if score > 60 else 0
assert grade_status == 1 or grade_status == 0

different: int = 1 if a != b else 0
assert different == 1 or different == 0

equal_check: int = 1 if a == b else 0
assert equal_check == 1 or equal_check == 0

# Access level checking
access_granted: int = 1 if level == 5 else 0
assert access_granted == 1 or access_granted == 0

# Complex symbolic conditions
special_category: int = 1 if (age >= 65 and income < 30000) else 0
assert special_category == 1 or special_category == 0

high_priority: int = 1 if (urgent or critical) else 0
assert high_priority == 1 or high_priority == 0

# Float-based symbolic operations
fever_check: float = 1.0 if temperature > 98.6 else 0.0
assert fever_check == 1.0 or fever_check == 0.0

bmi_valid: int = 1 if (weight > 0.0 and height > 0.0) else 0
assert bmi_valid == 1 or bmi_valid == 0

# Nested symbolic ternary
risk_level: int = 2 if age >= 65 else (1 if age >= 18 else 0)
assert risk_level >= 0 and risk_level <= 2

grade_category: int = 3 if score >= 90 else (2 if score >= 80 else (1 if score >= 70 else 0))
assert grade_category >= 0 and grade_category <= 3

# Boolean symbolic results
is_adult: bool = True if age >= 18 else False
assert is_adult == True or is_adult == False

is_eligible: bool = True if (age >= 21 and score >= 75) else False
assert is_eligible == True or is_eligible == False

# Mixed type promotion with symbolic values
priority_weight: float = 2.5 if urgent else 1  # int 1 promoted to float 1.0
assert priority_weight == 2.5 or priority_weight == 1.0

temperature_category: float = 3.0 if temperature > 100.0 else (2.0 if temperature > 98.6 else 1.0)
assert temperature_category >= 1.0 and temperature_category <= 3.0

# Edge case symbolic testing
zero_temperature: int = 1 if temperature == 0.0 else 0
assert zero_temperature == 1 or zero_temperature == 0

negative_score: int = 1 if score < 0 else 0
assert negative_score == 1 or negative_score == 0
