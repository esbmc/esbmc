class Decimal:
    _sign: int = 0
    _int: int = 0
    _exp: int = 0
    _is_special: int = 0

    def __init__(self, sign: int, int_val: int, exp: int, is_special: int):
        self._sign: int = sign
        self._int: int = int_val
        self._exp: int = exp
        self._is_special: int = is_special

    @staticmethod
    def _from_int(n: int) -> "Decimal":
        sign: int = 0
        val: int = n
        if n < 0:
            sign = 1
            val = 0 - n
        return Decimal(sign, val, 0, 0)

    def __eq__(self, other: "Decimal") -> bool:
        if self._is_special >= 2:
            return False
        if other._is_special >= 2:
            return False
        if self._is_special == 1 and other._is_special == 1:
            return self._sign == other._sign
        if self._is_special == 1 or other._is_special == 1:
            return False
        if self._int == 0 and other._int == 0:
            return True
        if self._sign != other._sign:
            return False
        self_int: int = self._int
        other_int: int = other._int
        self_exp: int = self._exp
        other_exp: int = other._exp
        while self_exp > other_exp:
            self_int = self_int * 10
            self_exp = self_exp - 1
        while other_exp > self_exp:
            other_int = other_int * 10
            other_exp = other_exp - 1
        return self_int == other_int

    def __lt__(self, other: "Decimal") -> bool:
        if self._is_special >= 2:
            return False
        if other._is_special >= 2:
            return False
        if self._is_special == 1 and other._is_special == 1:
            if self._sign == other._sign:
                return False
            return self._sign == 1
        if self._is_special == 1:
            return self._sign == 1
        if other._is_special == 1:
            return other._sign == 0
        if self._int == 0 and other._int == 0:
            return False
        if self._sign != other._sign:
            return self._sign == 1
        self_int: int = self._int
        other_int: int = other._int
        self_exp: int = self._exp
        other_exp: int = other._exp
        while self_exp > other_exp:
            self_int = self_int * 10
            self_exp = self_exp - 1
        while other_exp > self_exp:
            other_int = other_int * 10
            other_exp = other_exp - 1
        if self._sign == 1:
            return self_int > other_int
        return self_int < other_int

    def __le__(self, other: "Decimal") -> bool:
        if self._is_special >= 2:
            return False
        if other._is_special >= 2:
            return False
        if self._is_special == 1 and other._is_special == 1:
            if self._sign == other._sign:
                return True
            return self._sign == 1
        if self._is_special == 1:
            return self._sign == 1
        if other._is_special == 1:
            return other._sign == 0
        if self._int == 0 and other._int == 0:
            return True
        if self._sign != other._sign:
            return self._sign == 1
        self_int: int = self._int
        other_int: int = other._int
        self_exp: int = self._exp
        other_exp: int = other._exp
        while self_exp > other_exp:
            self_int = self_int * 10
            self_exp = self_exp - 1
        while other_exp > self_exp:
            other_int = other_int * 10
            other_exp = other_exp - 1
        if self._sign == 1:
            return self_int >= other_int
        return self_int <= other_int

    def __gt__(self, other: "Decimal") -> bool:
        if self._is_special >= 2:
            return False
        if other._is_special >= 2:
            return False
        if self._is_special == 1 and other._is_special == 1:
            if self._sign == other._sign:
                return False
            return self._sign == 0
        if self._is_special == 1:
            return self._sign == 0
        if other._is_special == 1:
            return other._sign == 1
        if self._int == 0 and other._int == 0:
            return False
        if self._sign != other._sign:
            return self._sign == 0
        self_int: int = self._int
        other_int: int = other._int
        self_exp: int = self._exp
        other_exp: int = other._exp
        while self_exp > other_exp:
            self_int = self_int * 10
            self_exp = self_exp - 1
        while other_exp > self_exp:
            other_int = other_int * 10
            other_exp = other_exp - 1
        if self._sign == 1:
            return self_int < other_int
        return self_int > other_int

    def __ge__(self, other: "Decimal") -> bool:
        if self._is_special >= 2:
            return False
        if other._is_special >= 2:
            return False
        if self._is_special == 1 and other._is_special == 1:
            if self._sign == other._sign:
                return True
            return self._sign == 0
        if self._is_special == 1:
            return self._sign == 0
        if other._is_special == 1:
            return other._sign == 1
        if self._int == 0 and other._int == 0:
            return True
        if self._sign != other._sign:
            return self._sign == 0
        self_int: int = self._int
        other_int: int = other._int
        self_exp: int = self._exp
        other_exp: int = other._exp
        while self_exp > other_exp:
            self_int = self_int * 10
            self_exp = self_exp - 1
        while other_exp > self_exp:
            other_int = other_int * 10
            other_exp = other_exp - 1
        if self._sign == 1:
            return self_int <= other_int
        return self_int >= other_int

    def __neg__(self) -> "Decimal":
        return Decimal(1 - self._sign, self._int, self._exp, self._is_special)

    def __abs__(self) -> "Decimal":
        return Decimal(0, self._int, self._exp, self._is_special)

    def __ne__(self, other: "Decimal") -> bool:
        if self._is_special >= 2:
            return True
        if other._is_special >= 2:
            return True
        if self._is_special == 1 and other._is_special == 1:
            return self._sign != other._sign
        if self._is_special == 1 or other._is_special == 1:
            return True
        if self._int == 0 and other._int == 0:
            return False
        if self._sign != other._sign:
            return True
        self_int: int = self._int
        other_int: int = other._int
        self_exp: int = self._exp
        other_exp: int = other._exp
        while self_exp > other_exp:
            self_int = self_int * 10
            self_exp = self_exp - 1
        while other_exp > self_exp:
            other_int = other_int * 10
            other_exp = other_exp - 1
        return self_int != other_int

    def __add__(self, other: "Decimal") -> "Decimal":
        # NaN propagation: sNaN first, then qNaN
        if self._is_special == 3 or other._is_special == 3:
            return Decimal(0, 0, 0, 2)
        if self._is_special == 2 or other._is_special == 2:
            return Decimal(0, 0, 0, 2)
        # Infinity handling
        if self._is_special == 1 and other._is_special == 1:
            if self._sign != other._sign:
                return Decimal(0, 0, 0, 2)  # Inf + (-Inf) = NaN
            return Decimal(self._sign, 0, 0, 1)
        if self._is_special == 1:
            return Decimal(self._sign, 0, 0, 1)
        if other._is_special == 1:
            return Decimal(other._sign, 0, 0, 1)
        # Finite addition: align exponents
        self_int: int = self._int
        other_int: int = other._int
        self_exp: int = self._exp
        other_exp: int = other._exp
        while self_exp > other_exp:
            self_int = self_int * 10
            self_exp = self_exp - 1
        while other_exp > self_exp:
            other_int = other_int * 10
            other_exp = other_exp - 1
        # Convert to signed values
        self_val: int = self_int
        if self._sign == 1:
            self_val = 0 - self_int
        other_val: int = other_int
        if other._sign == 1:
            other_val = 0 - other_int
        result_val: int = self_val + other_val
        result_sign: int = 0
        result_int: int = result_val
        if result_val < 0:
            result_sign = 1
            result_int = 0 - result_val
        return Decimal(result_sign, result_int, self_exp, 0)

    def __sub__(self, other: "Decimal") -> "Decimal":
        # NaN propagation: sNaN first, then qNaN
        if self._is_special == 3 or other._is_special == 3:
            return Decimal(0, 0, 0, 2)
        if self._is_special == 2 or other._is_special == 2:
            return Decimal(0, 0, 0, 2)
        # Flip other's sign
        neg_other_sign: int = 1 - other._sign
        # Infinity handling
        if self._is_special == 1 and other._is_special == 1:
            if self._sign != neg_other_sign:
                return Decimal(0, 0, 0, 2)  # Inf - Inf = NaN
            return Decimal(self._sign, 0, 0, 1)
        if self._is_special == 1:
            return Decimal(self._sign, 0, 0, 1)
        if other._is_special == 1:
            return Decimal(neg_other_sign, 0, 0, 1)
        # Finite subtraction: align exponents
        self_int: int = self._int
        other_int: int = other._int
        self_exp: int = self._exp
        other_exp: int = other._exp
        while self_exp > other_exp:
            self_int = self_int * 10
            self_exp = self_exp - 1
        while other_exp > self_exp:
            other_int = other_int * 10
            other_exp = other_exp - 1
        # Convert to signed values
        self_val: int = self_int
        if self._sign == 1:
            self_val = 0 - self_int
        other_val: int = other_int
        if neg_other_sign == 1:
            other_val = 0 - other_int
        result_val: int = self_val + other_val
        result_sign: int = 0
        result_int: int = result_val
        if result_val < 0:
            result_sign = 1
            result_int = 0 - result_val
        return Decimal(result_sign, result_int, self_exp, 0)

    def __mul__(self, other: "Decimal") -> "Decimal":
        # NaN propagation
        if self._is_special == 3 or other._is_special == 3:
            return Decimal(0, 0, 0, 2)
        if self._is_special == 2 or other._is_special == 2:
            return Decimal(0, 0, 0, 2)
        # Result sign = XOR of signs
        result_sign: int = 0
        if self._sign != other._sign:
            result_sign = 1
        # Infinity handling
        if self._is_special == 1 or other._is_special == 1:
            if self._int == 0 and self._is_special == 0:
                return Decimal(0, 0, 0, 2)  # Inf * 0 = NaN
            if other._int == 0 and other._is_special == 0:
                return Decimal(0, 0, 0, 2)  # 0 * Inf = NaN
            return Decimal(result_sign, 0, 0, 1)
        # Finite multiplication
        result_int: int = self._int * other._int
        result_exp: int = self._exp + other._exp
        return Decimal(result_sign, result_int, result_exp, 0)

    def __truediv__(self, other: "Decimal") -> "Decimal":
        # NaN propagation
        if self._is_special == 3 or other._is_special == 3:
            return Decimal(0, 0, 0, 2)
        if self._is_special == 2 or other._is_special == 2:
            return Decimal(0, 0, 0, 2)
        # Result sign
        result_sign: int = 0
        if self._sign != other._sign:
            result_sign = 1
        # Infinity / Infinity = NaN
        if self._is_special == 1 and other._is_special == 1:
            return Decimal(0, 0, 0, 2)
        # Inf / finite = Inf
        if self._is_special == 1:
            return Decimal(result_sign, 0, 0, 1)
        # finite / Inf = 0
        if other._is_special == 1:
            return Decimal(result_sign, 0, 0, 0)
        # 0 / 0 = NaN
        if self._int == 0 and other._int == 0:
            return Decimal(0, 0, 0, 2)
        # x / 0 = Inf
        if other._int == 0:
            return Decimal(result_sign, 0, 0, 1)
        # Scale numerator by 10^28
        numerator: int = self._int
        i: int = 0
        while i < 28:
            numerator = numerator * 10
            i = i + 1
        result_int: int = numerator // other._int
        result_exp: int = self._exp - other._exp - 28
        return Decimal(result_sign, result_int, result_exp, 0)

    def __floordiv__(self, other: "Decimal") -> "Decimal":
        # NaN propagation
        if self._is_special == 3 or other._is_special == 3:
            return Decimal(0, 0, 0, 2)
        if self._is_special == 2 or other._is_special == 2:
            return Decimal(0, 0, 0, 2)
        # Result sign
        result_sign: int = 0
        if self._sign != other._sign:
            result_sign = 1
        # Inf // Inf = NaN
        if self._is_special == 1 and other._is_special == 1:
            return Decimal(0, 0, 0, 2)
        # Inf // finite = Inf
        if self._is_special == 1:
            return Decimal(result_sign, 0, 0, 1)
        # finite // Inf = 0
        if other._is_special == 1:
            return Decimal(result_sign, 0, 0, 0)
        # 0 // 0 = NaN
        if self._int == 0 and other._int == 0:
            return Decimal(0, 0, 0, 2)
        # x // 0 = Inf
        if other._int == 0:
            return Decimal(result_sign, 0, 0, 1)
        # Align exponents
        self_int: int = self._int
        other_int: int = other._int
        self_exp: int = self._exp
        other_exp: int = other._exp
        while self_exp > other_exp:
            self_int = self_int * 10
            self_exp = self_exp - 1
        while other_exp > self_exp:
            other_int = other_int * 10
            other_exp = other_exp - 1
        quotient: int = self_int // other_int
        return Decimal(result_sign, quotient, 0, 0)

    def __mod__(self, other: "Decimal") -> "Decimal":
        # NaN propagation
        if self._is_special == 3 or other._is_special == 3:
            return Decimal(0, 0, 0, 2)
        if self._is_special == 2 or other._is_special == 2:
            return Decimal(0, 0, 0, 2)
        # Inf % x = NaN
        if self._is_special == 1:
            return Decimal(0, 0, 0, 2)
        # x % Inf = x
        if other._is_special == 1:
            return Decimal(self._sign, self._int, self._exp, 0)
        # x % 0 = NaN
        if other._int == 0:
            return Decimal(0, 0, 0, 2)
        # Align exponents
        self_int: int = self._int
        other_int: int = other._int
        self_exp: int = self._exp
        other_exp: int = other._exp
        while self_exp > other_exp:
            self_int = self_int * 10
            self_exp = self_exp - 1
        while other_exp > self_exp:
            other_int = other_int * 10
            other_exp = other_exp - 1
        quotient: int = self_int // other_int
        remainder: int = self_int - quotient * other_int
        return Decimal(self._sign, remainder, self_exp, 0)

    def __pos__(self) -> "Decimal":
        return Decimal(self._sign, self._int, self._exp, self._is_special)

    def is_nan(self) -> bool:
        return self._is_special >= 2

    def is_snan(self) -> bool:
        return self._is_special == 3

    def is_qnan(self) -> bool:
        return self._is_special == 2

    def is_infinite(self) -> bool:
        return self._is_special == 1

    def is_finite(self) -> bool:
        return self._is_special == 0

    def is_zero(self) -> bool:
        if self._is_special != 0:
            return False
        return self._int == 0

    def is_signed(self) -> bool:
        return self._sign == 1

    def copy_abs(self) -> "Decimal":
        return Decimal(0, self._int, self._exp, self._is_special)

    def copy_negate(self) -> "Decimal":
        return Decimal(1 - self._sign, self._int, self._exp, self._is_special)

    def copy_sign(self, other: "Decimal") -> "Decimal":
        return Decimal(other._sign, self._int, self._exp, self._is_special)

    def adjusted(self) -> int:
        if self._is_special != 0:
            return 0
        return _digit_count(self._int) - 1 + self._exp

    def is_normal(self) -> bool:
        if self._is_special != 0:
            return False
        if self._int == 0:
            return False
        return self.adjusted() >= -999999

    def is_subnormal(self) -> bool:
        if self._is_special != 0:
            return False
        if self._int == 0:
            return False
        return self.adjusted() < -999999

    def compare(self, other: "Decimal") -> "Decimal":
        if self._is_special >= 2 or other._is_special >= 2:
            return Decimal(0, 0, 0, 2)
        if self.__eq__(other):
            return Decimal(0, 0, 0, 0)
        if self.__lt__(other):
            return Decimal(1, 1, 0, 0)
        return Decimal(0, 1, 0, 0)

    def compare_signal(self, other: "Decimal") -> "Decimal":
        return self.compare(other)

    def max(self, other: "Decimal") -> "Decimal":
        if self._is_special >= 2 and other._is_special >= 2:
            return Decimal(0, 0, 0, 2)
        if self._is_special >= 2:
            return Decimal(other._sign, other._int, other._exp, other._is_special)
        if other._is_special >= 2:
            return Decimal(self._sign, self._int, self._exp, self._is_special)
        if self.__gt__(other):
            return Decimal(self._sign, self._int, self._exp, self._is_special)
        return Decimal(other._sign, other._int, other._exp, other._is_special)

    def min(self, other: "Decimal") -> "Decimal":
        if self._is_special >= 2 and other._is_special >= 2:
            return Decimal(0, 0, 0, 2)
        if self._is_special >= 2:
            return Decimal(other._sign, other._int, other._exp, other._is_special)
        if other._is_special >= 2:
            return Decimal(self._sign, self._int, self._exp, self._is_special)
        if self.__lt__(other):
            return Decimal(self._sign, self._int, self._exp, self._is_special)
        return Decimal(other._sign, other._int, other._exp, other._is_special)

    def fma(self, other: "Decimal", third: "Decimal") -> "Decimal":
        return self.__mul__(other).__add__(third)

    def normalize(self) -> "Decimal":
        if self._is_special == 3:
            return Decimal(0, 0, 0, 2)
        if self._is_special == 2:
            return Decimal(0, 0, 0, 2)
        if self._is_special == 1:
            return Decimal(self._sign, 0, 0, 1)
        if self._int == 0:
            return Decimal(self._sign, 0, 0, 0)
        coeff: int = self._int
        exp: int = self._exp
        i: int = 0
        while coeff > 0 and coeff % 10 == 0 and i < 28:
            coeff = coeff // 10
            exp = exp + 1
            i = i + 1
        return Decimal(self._sign, coeff, exp, 0)

    def to_integral_value(self) -> "Decimal":
        if self._is_special == 3:
            return Decimal(0, 0, 0, 2)
        if self._is_special == 2:
            return Decimal(0, 0, 0, 2)
        if self._is_special == 1:
            return Decimal(self._sign, 0, 0, 1)
        if self._exp >= 0:
            return Decimal(self._sign, self._int, self._exp, 0)
        divisor: int = 1
        neg_exp: int = 0 - self._exp
        i: int = 0
        while i < neg_exp and i < 28:
            divisor = divisor * 10
            i = i + 1
        result: int = _round_half_even(self._int, divisor)
        return Decimal(self._sign, result, 0, 0)

    def __float__(self) -> float:
        assert self._is_special == 0
        coeff: int = self._int
        exp: int = self._exp
        while exp > 0:
            coeff = coeff * 10
            exp = exp - 1
        fval: float = 0.0 + coeff
        while exp < 0:
            fval = fval / 10.0
            exp = exp + 1
        if self._sign == 1:
            fval = 0.0 - fval
        return fval

    def quantize(self, other: "Decimal") -> "Decimal":
        if self._is_special == 3 or other._is_special == 3:
            return Decimal(0, 0, 0, 2)
        if self._is_special == 2 or other._is_special == 2:
            return Decimal(0, 0, 0, 2)
        if self._is_special == 1 and other._is_special == 1:
            return Decimal(self._sign, 0, 0, 1)
        if self._is_special == 1:
            return Decimal(0, 0, 0, 2)
        if other._is_special == 1:
            return Decimal(0, 0, 0, 2)
        target_exp: int = other._exp
        coeff: int = self._int
        cur_exp: int = self._exp
        if target_exp < cur_exp:
            i: int = 0
            diff: int = cur_exp - target_exp
            while i < diff and i < 28:
                coeff = coeff * 10
                i = i + 1
        elif target_exp > cur_exp:
            divisor: int = 1
            diff2: int = target_exp - cur_exp
            j: int = 0
            while j < diff2 and j < 28:
                divisor = divisor * 10
                j = j + 1
            coeff = _round_half_even(coeff, divisor)
        return Decimal(self._sign, coeff, target_exp, 0)

    def sqrt(self) -> "Decimal":
        if self._is_special == 3 or self._is_special == 2:
            return Decimal(0, 0, 0, 2)
        if self._is_special == 1:
            if self._sign == 1:
                return Decimal(0, 0, 0, 2)
            return Decimal(0, 0, 0, 1)
        if self._int == 0:
            return Decimal(0, 0, 0, 0)
        if self._sign == 1:
            return Decimal(0, 0, 0, 2)
        prec: int = 28
        coeff: int = self._int
        adj_exp: int = self._exp
        # Scale up by 2*prec digits for precision
        i: int = 0
        while i < 2 * prec:
            coeff = coeff * 10
            adj_exp = adj_exp - 1
            i = i + 1
        # Ensure adjusted exponent is even
        abs_exp: int = adj_exp
        if abs_exp < 0:
            abs_exp = 0 - abs_exp
        if abs_exp % 2 == 1:
            coeff = coeff * 10
            adj_exp = adj_exp - 1
        # Newton's method for integer square root
        dc: int = _digit_count(coeff)
        x: int = 1
        k: int = 0
        half_dc: int = (dc + 1) // 2
        while k < half_dc:
            x = x * 10
            k = k + 1
        it: int = 0
        while it < 100:
            x_new: int = (x + coeff // x) // 2
            if x_new >= x:
                it = 100
            else:
                x = x_new
            it = it + 1
        result_exp: int = adj_exp // 2
        return Decimal(0, x, result_exp, 0)

    def __bool__(self) -> bool:
        if self._is_special != 0:
            return True
        return self._int != 0

    def __int__(self) -> int:
        assert self._is_special == 0
        result: int = self._int
        exp: int = self._exp
        while exp > 0:
            result = result * 10
            exp = exp - 1
        while exp < 0:
            result = result // 10
            exp = exp + 1
        if self._sign == 1:
            result = 0 - result
        return result

    def __radd__(self, other: int) -> "Decimal":
        return self.__add__(_decimal_from_int(other))

    def __rsub__(self, other: int) -> "Decimal":
        return _decimal_from_int(other).__sub__(self)

    def __rmul__(self, other: int) -> "Decimal":
        return self.__mul__(_decimal_from_int(other))

    def __rtruediv__(self, other: int) -> "Decimal":
        return _decimal_from_int(other).__truediv__(self)

    def __rfloordiv__(self, other: int) -> "Decimal":
        return _decimal_from_int(other).__floordiv__(self)

    def __rmod__(self, other: int) -> "Decimal":
        return _decimal_from_int(other).__mod__(self)


def _digit_count(n: int) -> int:
    if n == 0:
        return 1
    count: int = 0
    val: int = n
    while val > 0:
        val = val // 10
        count = count + 1
    return count


def _decimal_from_int(n: int) -> "Decimal":
    sign: int = 0
    val: int = n
    if n < 0:
        sign = 1
        val = 0 - n
    return Decimal(sign, val, 0, 0)


def _round_half_even(coeff: int, divisor: int) -> int:
    quotient: int = coeff // divisor
    remainder: int = coeff - quotient * divisor
    half: int = divisor // 2
    if remainder > half:
        quotient = quotient + 1
    elif remainder == half:
        if quotient % 2 == 1:
            quotient = quotient + 1
    return quotient
