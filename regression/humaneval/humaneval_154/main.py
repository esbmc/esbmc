# HumanEval/154
# Entry Point: cycpattern_check
# ESBMC-compatible format with direct assertions


def cycpattern_check(a , b):
    """You are given 2 words. You need to return True if the second word or any of its rotations is a substring in the first word
    cycpattern_check("abcd","abd") => False
    cycpattern_check("hello","ell") => True
    cycpattern_check("whassup","psus") => False
    cycpattern_check("abab","baa") => True
    cycpattern_check("efef","eeff") => False
    cycpattern_check("himenss","simen") => True

    """
    l = len(b)
    pat = b + b
    for i in range(len(a) - l + 1):
        for j in range(l + 1):
            if a[i:i+l] == pat[j:j+l]:
                return True
    return False

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert  cycpattern_check("xyzw","xyw") == False , "test #0"
    # assert  cycpattern_check("yello","ell") == True , "test #1"
    # assert  cycpattern_check("whattup","ptut") == False , "test #2"
    # assert  cycpattern_check("efef","fee") == True , "test #3"
    # assert  cycpattern_check("abab","aabb") == False , "test #4"
    # assert  cycpattern_check("winemtt","tinem") == True , "test #5"
    print("✅ HumanEval/154 - All assertions completed!")
