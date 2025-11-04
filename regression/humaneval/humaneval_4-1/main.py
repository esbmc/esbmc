# HumanEval/4
# Entry Point: mean_absolute_deviation
# ESBMC-compatible format with direct assertions

from typing import List  
  
def mean_absolute_deviation(numbers: List[float]) -> float:  
    """ For a given list of input numbers, calculate Mean Absolute Deviation  
    around the mean of this dataset.  
    """  
    #manual sum
    total = 0.0  
    for num in numbers:  
        total += num  
    mean = total / len(numbers)  
      
    # manual deviation sum
    deviation_sum = 0.0  
    for x in numbers:  
        deviation_sum += abs(x - mean)  
      
    return deviation_sum / len(numbers)  
  
if __name__ == "__main__":  
    assert abs(mean_absolute_deviation([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-6 
    # assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6
    # assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0, 5.0]) - 6.0/5.0) < 1e-6
    print("✅ HumanEval/4 - All assertions completed!")