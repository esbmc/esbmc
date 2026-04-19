from typing import List

def annotated_list() -> List[int]:
    return [1, 2, 3]

result = annotated_list()
assert result[0] != 1
assert result[1] != 2
assert result[2] != 3
