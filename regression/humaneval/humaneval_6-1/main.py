# HumanEval/6  
# Entry Point: parse_nested_parens  
# ESBMC-compatible format with direct assertions  
  
from typing import List  
  
  
def parse_paren_group(s: str) -> int:  
    """Helper function to calculate max nesting depth for a single group"""  
    depth: int = 0  
    max_depth: int = 0  
    i: int = 0  
    while i < len(s):  
        c: str = s[i]  
        if c == '(':  
            depth = depth + 1  
            if depth > max_depth:  
                max_depth = depth  
        else:  
            depth = depth - 1  
        i = i + 1  
    return max_depth  
  
  
def parse_nested_parens(paren_string: str) -> List[int]:  
    """ Input to this function is a string represented multiple groups for nested parentheses separated by spaces.  
    For each of the group, output the deepest level of nesting of parentheses.  
    E.g. (()()) has maximum two levels of nesting while ((())) has three.  
  
    >>> parse_nested_parens('(()()) ((())) () ((())()())')  
    [2, 3, 1, 3]  
    """  
    result: List[int] = []  
    current_group: str = ""  
    i: int = 0  
      
    # Manually parse groups separated by spaces  
    while i < len(paren_string):  
        c: str = paren_string[i]  
        if c == ' ':  
            if len(current_group) > 0:  
                depth_val: int = parse_paren_group(current_group)  
                result.append(depth_val)  
                current_group = ""  
        else:  
            current_group = current_group + c  
        i = i + 1  
      
    # Don't forget the last group  
    if len(current_group) > 0:  
        depth_val: int = parse_paren_group(current_group)  
        result.append(depth_val)  
      
    return result  
  
  
# Direct assertions for ESBMC verification  
if __name__ == "__main__":  
    result1: List[int] = parse_nested_parens('(()()) ((())) () ((())()())')  
    assert result1[0] == 2  
    assert result1[1] == 3  
    assert result1[2] == 1  
    assert result1[3] == 3  
      
    print("✅ HumanEval/6 - All assertions completed!")

