from typing import Literal


def test_single_string(s: Literal["foo"]) -> int:
    return 42


from typing import Literal


def test_single_int(x: Literal[42]) -> int:
    return x


from typing import Literal


def test_single_bool(b: Literal[True]) -> bool:
    return b


from typing import Literal


def test_single_none(n: Literal[None]) -> int:
    return 0


from typing import Literal


def test_multi_int(x: Literal[1, 2, 3]) -> int:
    return x


from typing import Literal


def test_multi_string(s: Literal["a", "b", "c"]) -> str:
    return s


from typing import Literal


def test_multi_bool(b: Literal[True, False]) -> bool:
    return b


from typing import Literal


def test_multi_float(f: Literal[1.0, 2.5, 3.14]) -> float:
    return f
