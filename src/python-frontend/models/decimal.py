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
