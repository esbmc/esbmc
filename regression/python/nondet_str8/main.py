# Test nondet_str() with function parameter passing

def check_length(s: str) -> int:
    return len(s)

result = check_length(nondet_str())

# The length of any string is always >= 0
assert result >= 0

