def classify_number(n: int) -> str:
    """Classify a number and return appropriate message"""
    if n > 0:
        if n > 100:
            return "large positive"
        else:
            return "small positive"
    elif n < 0:
        if n < -100:
            return "large negative"
        else:
            return "small negative"
    else:
        return "zero"

# Test generation mode: use nondet inputs
result = classify_number(__VERIFIER_nondet_int())

