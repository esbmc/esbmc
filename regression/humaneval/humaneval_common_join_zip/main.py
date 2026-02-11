# Common join test via zip + list comprehension
from typing import List


def xor_join(a: str, b: str) -> str:
    result: List[str] = []
    i: int = 0
    len_a: int = len(a)
    len_b: int = len(b)
    n: int = len_a if len_a < len_b else len_b

    while i < n:
        x = a[i]
        y = b[i]
        if x == y:
            result.append('0')
        else:
            result.append('1')
        i = i + 1
    return ''.join(result)


if __name__ == "__main__":
    assert xor_join('111000', '101010') == '010010'
    print("ok")
