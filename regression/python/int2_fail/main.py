def test_invalid_word():
    """Invalid string literal - should fail with ValueError"""
    x = int("user_input")

def test_invalid_mixed_alphanumeric():
    """String with numbers and letters - should fail"""
    y = int("123abc")

def test_invalid_letters_first():
    """String with letters followed by numbers - should fail"""
    z = int("abc123")

def test_invalid_empty_string():
    """Empty string - should fail"""
    a = int("")

def test_invalid_sign_only_plus():
    """String with only plus sign - should fail"""
    b = int("+")

def test_invalid_sign_only_minus():
    """String with only minus sign - should fail"""
    c = int("-")

def test_invalid_whitespace_only():
    """String with only whitespace - should fail"""
    d = int(" ")

def test_invalid_with_whitespace():
    """Valid number with whitespace - should fail (no trimming in minimal fix)"""
    e = int(" 123 ")

def test_invalid_hex_string():
    """Hexadecimal string - should fail (only decimal supported)"""
    f = int("0x10")

def test_invalid_binary_string():
    """Binary string - should fail"""
    g = int("0b1010")

def test_invalid_octal_string():
    """Octal string - should fail"""
    h = int("0o777")

def test_invalid_float_string():
    """Float string - should fail"""
    i = int("12.34")

def test_invalid_scientific_notation():
    """Scientific notation - should fail"""
    j = int("1e5")

def test_invalid_special_chars():
    """String with special characters - should fail"""
    k = int("!@#$")

def test_invalid_words():
    """Common English words - should fail"""
    l = int("hello")
    m = int("world")

def test_invalid_double_sign():
    """String with double signs - should fail"""
    n = int("++123")
    o = int("--456")

def test_invalid_sign_at_end():
    """String with sign at end - should fail"""
    p = int("123+")
    q = int("456-")

def test_invalid_mixed_signs():
    """String with mixed signs - should fail"""
    r = int("+-789")

def test_invalid_internal_sign():
    """String with sign in middle - should fail"""
    s = int("12+34")

def test1(): x = int("invalid")
def test2(): x = int("123abc") 
def test3(): x = int("")
def test4(): x = int("+")
def test5(): x = int("-")
def test6(): x = int(" 123 ")
def test7(): x = int("0x10")
def test8(): x = int("12.34")
def test9(): x = int("hello")
def test10(): x = int("!@#$")

if __name__ == "__main__":
    test_invalid_word()
    test_invalid_mixed_alphanumeric()
    test_invalid_empty_string()
    test_invalid_sign_only_plus()
    test_invalid_with_whitespace()
    test_invalid_hex_string()
    test_invalid_float_string()
    test_invalid_words()
    test1()  # Simple isolated test

