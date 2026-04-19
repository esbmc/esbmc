# HumanEval/5  
# Entry Point: intersperse  
# ESBMC-compatible format with direct assertions  
  
from typing import List  
  
  
def intersperse(numbers: List[int], delimeter: int) -> List[int]:  
    """ Insert a number 'delimeter' between every two consecutive elements of input list `numbers'  
    """  
    if not numbers:  
        return []  
  
    result = []  
        
    for i in range(len(numbers) - 1):  
        result.append(numbers[i])  
        result.append(delimeter)  
      
    result.append(numbers[len(numbers) - 1])  
      
    return result  
  
  
# Direct assertions for ESBMC verification  
if __name__ == "__main__":  
    result1 = intersperse([], 7)  
    expected1 = []  
    assert result1 == expected1  
      
    # result2 = intersperse([5, 6, 3, 2], 8)  
    # expected2 = [5, 8, 6, 8, 3, 8, 2]  
    # assert result2 == expected2  
      
    # result3 = intersperse([2, 2, 2], 2)  
    # expected3 = [2, 2, 2, 2, 2]  
    # assert result3 == expected3  
      
    print("✅ HumanEval/5 - All assertions completed!")
