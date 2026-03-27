# Test list comprehension with function call - FAIL case
# Verifies that incorrect assertions are detected

def increment(x: int) -> int:
    return x + 1

if __name__ == "__main__":
    nums = [1, 2, 3]
    
    res = [increment(x) for x in nums]
    assert res[0] == 99
