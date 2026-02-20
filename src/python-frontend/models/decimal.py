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
