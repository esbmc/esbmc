# Debug 4: original algorithm with split
from typing import List

def parse_nested_parens(paren_string: str) -> List[int]:
    def parse_paren_group(s: str) -> int:
        depth = 0
        max_depth = 0
        for c in s:
            if c == '(':
                depth += 1
                if depth > max_depth:
                    max_depth = depth
            else:
                depth -= 1
        return max_depth

    return [parse_paren_group(x) for x in paren_string.split(' ') if x]

if __name__ == "__main__":
    result: List[int] = parse_nested_parens('(()()) ((())) () ((())()())')
    assert result == [2, 3, 1, 3]
