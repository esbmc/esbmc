def calculate_grade(score: int) -> str:
    """Function with missing return statement for some execution paths"""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    # Missing return statement for score < 60!


def process_number(x: int) -> int:
    """Another function with missing return for positive numbers"""
    if x == 0:
        return 0
    if x < 0:
        return -x
    # Missing return for positive numbers - just has expression without return
    x * 2


def safe_divide(a: int, b: int) -> int:
    """Correctly implemented function with all paths covered"""
    if b == 0:
        return 0
    else:
        return a // b


# Test the functions
score = 50
result = calculate_grade(score)  # This will trigger missing return detection

value = process_number(5)  # This will also trigger missing return detection

safe_result = safe_divide(10, 2)  # This should work fine
