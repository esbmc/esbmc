def factorial(n: int) -> int:
    assert n >= 0
    assert n <= 10
    
    if n <= 1:
        return 1
    else:
        result: int = n * factorial(n - 1)
        assert result > 0
        return result

def divide_with_checks(a: int, b: int) -> int:
    assert b != 0
    assert a >= 0
    assert b > 0
    
    result: int = a // b
    assert result >= 0
    
    remainder: int = a % b
    assert remainder >= 0 and remainder < b
    
    return result

def power_with_checks(base: int, exp: int) -> int:
    assert exp >= 0
    assert base != 0 or exp > 0
    
    if exp == 0:
        return 1
    
    result: int = 1
    count: int = 0
    while count < exp:
        assert count >= 0
        assert count < exp
        
        old_result: int = result
        result = result * base
        
        if base > 1:
            assert result > old_result
        
        count = count + 1
    
    assert result > 0 or base == 0
    return result

def gcd_with_checks(a: int, b: int) -> int:
    assert a > 0
    assert b > 0
    
    while b != 0:
        assert a > 0 and b > 0
        
        temp: int = b
        b = a % b
        a = temp
        
        assert b >= 0
    
    assert a > 0
    return a

# Test factorial with different values
factorial(0)
factorial(1)
factorial(5)

# Test division with checks
divide_with_checks(10, 2)
divide_with_checks(17, 5)
divide_with_checks(100, 7)

# Test power with checks
power_with_checks(2, 0)
power_with_checks(2, 3)
power_with_checks(3, 4)
power_with_checks(1, 10)

# Test gcd with checks
gcd_with_checks(48, 18)
gcd_with_checks(100, 50)
gcd_with_checks(7, 3)
