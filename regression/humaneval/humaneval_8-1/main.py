# HumanEval/8  
# Entry Point: sum_product  
# ESBMC-compatible format with direct assertions  
  
from typing import List  
  
  
def sum_product(numbers: List[int]) -> List[int]:  
    """ For a given list of integers, return a list with sum and product of all the integers.  
    Empty sum should be equal to 0 and empty product should be equal to 1.  
    >>> sum_product([])  
    [0, 1]  
    >>> sum_product([1, 2, 3, 4])  
    [10, 24]  
    """  
    sum_value: int = 0  
    prod_value: int = 1  
    i: int = 0  
  
    while i < len(numbers):  
        n: int = numbers[i]  
        sum_value = sum_value + n  
        prod_value = prod_value * n  
        i = i + 1  
      
    result: List[int] = []  
    result.append(sum_value)  
    result.append(prod_value)  
    return result  
  
  
# Direct assertions for ESBMC verification  
if __name__ == "__main__":  
    result1: List[int] = sum_product([])  
    assert result1[0] == 0  
    assert result1[1] == 1  
      
    result2: List[int] = sum_product([1, 1, 1])  
    assert result2[0] == 3  
    assert result2[1] == 1  
      
    result3: List[int] = sum_product([100, 0])  
    assert result3[0] == 100  
    assert result3[1] == 0  
      
    result4: List[int] = sum_product([3, 5, 7])  
    assert result4[0] == 15  
    assert result4[1] == 105  
      
    result5: List[int] = sum_product([10])  
    assert result5[0] == 10  
    assert result5[1] == 10  
      
    print("✅ HumanEval/8 - All assertions completed!")
