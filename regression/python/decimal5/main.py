from decimal import Decimal

def check_is_nan(d: Decimal) -> bool:
    return d.is_nan()

def check_is_snan(d: Decimal) -> bool:
    return d.is_snan()

def check_is_qnan(d: Decimal) -> bool:
    return d.is_qnan()

def check_is_infinite(d: Decimal) -> bool:
    return d.is_infinite()

def check_is_finite(d: Decimal) -> bool:
    return d.is_finite()

def check_is_zero(d: Decimal) -> bool:
    return d.is_zero()

def check_is_signed(d: Decimal) -> bool:
    return d.is_signed()

def get_sign(d: Decimal) -> int:
    return d._sign

def get_int(d: Decimal) -> int:
    return d._int

def get_exp(d: Decimal) -> int:
    return d._exp

def get_is_special(d: Decimal) -> int:
    return d._is_special

# is_nan
nan: Decimal = Decimal("NaN")
snan: Decimal = Decimal("sNaN")
inf: Decimal = Decimal("Infinity")
zero: Decimal = Decimal("0")
val: Decimal = Decimal("3.14")

assert check_is_nan(nan) == True
assert check_is_nan(snan) == True
assert check_is_nan(inf) == False
assert check_is_nan(zero) == False
assert check_is_nan(val) == False

# is_snan
assert check_is_snan(snan) == True
assert check_is_snan(nan) == False
assert check_is_snan(val) == False

# is_qnan
assert check_is_qnan(nan) == True
assert check_is_qnan(snan) == False
assert check_is_qnan(val) == False

# is_infinite
assert check_is_infinite(inf) == True
neg_inf: Decimal = Decimal("-Infinity")
assert check_is_infinite(neg_inf) == True
assert check_is_infinite(nan) == False
assert check_is_infinite(val) == False

# is_finite
assert check_is_finite(val) == True
assert check_is_finite(zero) == True
assert check_is_finite(inf) == False
assert check_is_finite(nan) == False

# is_zero
assert check_is_zero(zero) == True
neg_zero: Decimal = Decimal("-0")
assert check_is_zero(neg_zero) == True
assert check_is_zero(val) == False
assert check_is_zero(inf) == False
assert check_is_zero(nan) == False

# is_signed
neg_val: Decimal = Decimal("-3.14")
assert check_is_signed(neg_val) == True
assert check_is_signed(neg_inf) == True
assert check_is_signed(neg_zero) == True
assert check_is_signed(val) == False
assert check_is_signed(zero) == False
assert check_is_signed(inf) == False

# __pos__
pos_val: Decimal = +val
assert get_sign(pos_val) == 0
assert get_int(pos_val) == 314
assert get_exp(pos_val) == -2
assert get_is_special(pos_val) == 0

pos_neg: Decimal = +neg_val
assert get_sign(pos_neg) == 1
assert get_int(pos_neg) == 314
