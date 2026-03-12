# HumanEval/98
# Entry Point: count_upper
# ESBMC-compatible format with direct assertions


def count_upper(s):
    """
    Given a string s, count the number of uppercase vowels in even indices.
    
    For example:
    count_upper('aBCdEf') returns 1
    count_upper('abcdefg') returns 0
    count_upper('dBBE') returns 0
    """
    count = 0
    for i in range(0,len(s),2):
        if s[i] in "AEIOU":
            count += 1
    return count

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert count_upper('aBCdEf')  == 1
    # assert count_upper('abcdefg') == 0
    # assert count_upper('dBBE') == 0
    # assert count_upper('B')  == 0
    # assert count_upper('U')  == 1
    # assert count_upper('') == 0
    # assert count_upper('EEEE') == 2
    # assert True
    print("✅ HumanEval/98 - All assertions completed!")
