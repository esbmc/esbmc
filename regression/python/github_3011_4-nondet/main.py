from typing import List

def foo(arr: List[int]) -> None:
    if len(arr) <= 1:
        for i in arr:
            pass

arr = [nondet_int()]
foo(arr)
