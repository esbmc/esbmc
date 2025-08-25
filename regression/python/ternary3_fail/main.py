# Test Case 1: Integer comparisons
age: int = 20
status: str = "adult" if age >= 18 else "minor"
assert status == "adult"

age: int = 16  
status: str = "adult" if age >= 18 else "minor"
assert status == "minor"

name: str = "Bob"
greeting: str = "match" if name == "Alice" else "stranger" 
assert greeting == "stranger"

# Test Case 3: Boolean variables
is_enabled: bool = True
power: str = "on" if is_enabled else "off"
assert power == "on"

is_enabled: bool = False
power: str = "on" if is_enabled else "off"
assert power == "off"

# Test Case 4: Boolean equality comparisons  
flag1: bool = True
flag2: bool = True
comparison: str = "same" if flag1 == flag2 else "different"
assert comparison == "same"

flag1: bool = True
flag2: bool = False
comparison: str = "same" if flag1 == flag2 else "different"
assert comparison == "different"

# Test Case 5: All comparison operators with integers
x: int = 10
y: int = 5

result: str = "greater" if x > y else "not greater"
assert result == "greater"

result: str = "less" if x < y else "not less" 
assert result == "not less"

result: str = "equal" if x == y else "not equal"
assert result == "equal"

result: str = "not equal" if x != y else "equal"
assert result == "not equal"

result: str = "gte" if x >= y else "not gte"
assert result == "gte"

result: str = "lte" if x <= y else "not lte"
assert result == "not lte"

# Test Case 6: String ordering
str1: str = "apple"
str2: str = "zebra"

# Test Case 7: Edge cases
# Zero comparison
count: int = 0
message: str = "empty" if count == 0 else "not empty"
assert message == "empty"
