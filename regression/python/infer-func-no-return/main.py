# ====================================
# Functions without return annotations
# ====================================
def get_integer():
    return 1

def get_float():
    return 0.5

def get_boolean():
    return True

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

