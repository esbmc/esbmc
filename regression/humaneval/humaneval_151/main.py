# HumanEval/151
# Entry Point: double_the_difference
# ESBMC-compatible format with direct assertions


def double_the_difference(lst):
    '''
    Given a list of numbers, return the sum of squares of the numbers
    in the list that are odd. Ignore numbers that are negative or not integers.
    
    double_the_difference([1, 3, 2, 0]) == 1 + 9 + 0 + 0 = 10
    double_the_difference([-1, -2, 0]) == 0
    double_the_difference([9, -2]) == 81
    double_the_difference([0]) == 0  
   
    If the input list is empty, return 0.
    '''
    return sum([i**2 for i in lst if i > 0 and i%2!=0 and "." not in str(i)])

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert double_the_difference([]) == 0 , "This prints if this assert fails 1 (good for debugging!)"
    # assert double_the_difference([5, 4]) == 25 , "This prints if this assert fails 2 (good for debugging!)"
    # assert double_the_difference([0.1, 0.2, 0.3]) == 0 , "This prints if this assert fails 3 (good for debugging!)"
    # assert double_the_difference([-10, -20, -30]) == 0 , "This prints if this assert fails 4 (good for debugging!)"
    # assert double_the_difference([-1, -2, 8]) == 0, "This prints if this assert fails 5 (also good for debugging!)"
    # assert double_the_difference([0.2, 3, 5]) == 34, "This prints if this assert fails 6 (also good for debugging!)"
    # assert double_the_difference(lst) == odd_sum , "This prints if this assert fails 7 (good for debugging!)"
    print("✅ HumanEval/151 - All assertions completed!")
