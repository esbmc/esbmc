# ===============================================
# Functions with wrong return annotations
# to test type inference
# ===============================================

def int_float() -> int:
  return 0.5


def float_bool() -> float:
  return False


def bool_int() -> bool:
  return 3


# ==============================
# Test Cases with Assertions
# ==============================

# With incorrect annotations
d = int_float()
assert d == 0.5

e = float_bool()
assert e == False

f = bool_int()
assert f == 3
