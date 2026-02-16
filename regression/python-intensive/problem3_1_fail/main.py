from typing import List

def findWordsContaining_v1(words: List[str], x: str) -> List[int]:
    RES = []
    for i, el in enumerate(words):
        if x in el:
            RES.append(i)
    return RES

assert findWordsContaining_v1(["test"], "test") == [1, 2, 3, 4]
