# HumanEval/85
# Entry Point: add
# ESBMC-compatible format with direct assertions


def add(lst):
    """Given a non-empty list of integers lst. add the even elements that are at odd indices..


    Examples:
        add([4, 2, 6, 7]) ==> 2 
    """
    return sum([lst[i] for i in range(1, len(lst), 2) if lst[i]%2 == 0])

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert add([4, 88]) == 88
    # assert add([4, 5, 6, 7, 2, 122]) == 122
    # assert add([4, 0, 6, 7]) == 0
    # assert add([4, 4, 6, 8]) == 12
    print("✅ HumanEval/85 - All assertions completed!")
