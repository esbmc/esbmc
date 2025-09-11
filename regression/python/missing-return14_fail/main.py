import random

def lambda_test(x: int) -> int:
    # Lambda that might return None
    process = lambda n: n * 2 if n > 0 else None
    
    if x > 0:
        result = process(x)
        return result if result is not None else 0
    else:
        return -1

x = random.randint(0,10)
result = lambda_test(x)
