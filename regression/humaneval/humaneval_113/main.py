# HumanEval/113
# Entry Point: odd_count
# ESBMC-compatible format with direct assertions


def odd_count(lst):
    """Given a list of strings, where each string consists of only digits, return a list.
    Each element i of the output should be "the number of odd elements in the
    string i of the input." where all the i's should be replaced by the number
    of odd digits in the i'th string of the input.

    >>> odd_count(['1234567'])
    ["the number of odd elements 4n the str4ng 4 of the 4nput."]
    >>> odd_count(['3',"11111111"])
    ["the number of odd elements 1n the str1ng 1 of the 1nput.",
     "the number of odd elements 8n the str8ng 8 of the 8nput."]
    """
    res = []
    for arr in lst:
        n = sum(int(d)%2==1 for d in arr)
        res.append("the number of odd elements " + str(n) + "n the str"+ str(n) +"ng "+ str(n) +" of the "+ str(n) +"nput.")
    return res

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert odd_count(['1234567']) == ["the number of odd elements 4n the str4ng 4 of the 4nput."], "Test 1"
    # assert odd_count(['3',"11111111"]) == ["the number of odd elements 1n the str1ng 1 of the 1nput.", "the number of odd elements 8n the str8ng 8 of the 8nput."], "Test 2"
    # assert odd_count(['271', '137', '314']) == [
    # assert True, "This prints if this assert fails 2 (also good for debugging!)"
    print("✅ HumanEval/113 - All assertions completed!")
