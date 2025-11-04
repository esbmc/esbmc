# HumanEval/9  
# Entry Point: rolling_max  
# ESBMC-compatible format with direct assertions  
  
from typing import List  
  
  
def rolling_max(numbers: List[int]) -> List[int]:  
    """ From a given list of integers, generate a list of rolling maximum element found until given moment  
    in the sequence.  
    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])  
    [1, 2, 3, 3, 3, 4, 4]  
    """  
    result: List[int] = []  
      
    # Handle empty list case  
    if len(numbers) == 0:  
        return result  
      
    running_max: int = numbers[0]  
    result.append(running_max)  
      
    i: int = 1  
    while i < len(numbers):  
        n: int = numbers[i]  
        # Manual max implementation  
        if n > running_max:  
            running_max = n  
        result.append(running_max)  
        i = i + 1  
      
    return result  
  
  
# Direct assertions for ESBMC verification  
if __name__ == "__main__":  
    # Test empty list  
    result1: List[int] = rolling_max([])  
    assert len(result1) == 0  
      
    # Test ascending sequence  
    result2: List[int] = rolling_max([1, 2, 3, 4])  
    assert len(result2) == 4  
    assert result2[0] == 1  
    assert result2[1] == 2  
    assert result2[2] == 3  
    assert result2[3] == 4  
      
    # Test descending sequence  
    result3: List[int] = rolling_max([4, 3, 2, 1])  
    assert len(result3) == 4  
    assert result3[0] == 4  
    assert result3[1] == 4  
    assert result3[2] == 4  
    assert result3[3] == 4  
      
    # Test mixed sequence  
    result4: List[int] = rolling_max([3, 2, 3, 100, 3])  
    assert len(result4) == 5  
    assert result4[0] == 3  
    assert result4[1] == 3  
    assert result4[2] == 3  
    assert result4[3] == 100  
    assert result4[4] == 100  
      
    print("✅ HumanEval/9 - All assertions completed!")
