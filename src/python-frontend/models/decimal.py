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
