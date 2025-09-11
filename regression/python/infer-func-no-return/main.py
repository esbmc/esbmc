# ====================================
# Functions without return annotations
# ====================================
def get_integer():
    return 1


def get_float():
    return 0.5


def get_boolean():
    return True


# ===============================================
# Functions with wrong return annotations
# to test type inference
# ===============================================


def int_float() -> int:  # Should infer to float
    return 0.5


def float_bool() -> float:  # Should infer to bool
    return False


def bool_int() -> bool:  # Should infer to int
    return 3


# ==============================
# Test Cases with Assertions
# ==============================

# Without annotations
a = get_integer()
assert a == 1

b = get_float()
assert b == 0.5

c = get_boolean()
assert c == True

# With incorrect annotations
d = int_float()
assert d == 0.5

e = float_bool()
assert e == False

f = bool_int()
assert f == 3
