# HumanEval/7  
# Entry Point: filter_by_substring  
# ESBMC-compatible format with direct assertions  
  
from typing import List  
  
  
def contains_substring(text: str, substring: str) -> bool:  
    """Check if text contains substring using manual search"""  
    if len(substring) == 0:  
        return True  
      
    if len(text) < len(substring):  
        return False  
      
    i: int = 0  
    while i <= len(text) - len(substring):  
        j: int = 0  
        match: bool = True  
        while j < len(substring):  
            if text[i + j] != substring[j]:  
                match = False  
                break  
            j = j + 1  
        if match:  
            return True  
        i = i + 1  
      
    return False  
  
  
def filter_by_substring(strings: List[str], substring: str) -> List[str]:  
    """ Filter an input list of strings only for ones that contain given substring  
    >>> filter_by_substring([], 'a')  
    []  
    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')  
    ['abc', 'bacd', 'array']  
    """  
    result: List[str] = []  
    i: int = 0  
      
    while i < len(strings):  
        current_string: str = strings[i]  
        if contains_substring(current_string, substring):  
            result.append(current_string)  
        i = i + 1  
      
    return result  
  
  
# Direct assertions for ESBMC verification  
if __name__ == "__main__":  
    result1: List[str] = filter_by_substring([], 'john')  
    assert len(result1) == 0  
      
    result2: List[str] = filter_by_substring(['xxx', 'asd', 'xxy', 'john doe', 'xxxAAA', 'xxx'], 'xxx')  
    assert len(result2) == 3  
    assert result2[0] == 'xxx'  
    assert result2[1] == 'xxxAAA'  
    assert result2[2] == 'xxx'  
      
    print("✅ HumanEval/7 - All assertions completed!")
