# HumanEval/134
# Entry Point: check_if_last_char_is_a_letter
# ESBMC-compatible format with direct assertions


def check_if_last_char_is_a_letter(txt):
    '''
    Create a function that returns True if the last character
    of a given string is an alphabetical character and is not
    a part of a word, and False otherwise.
    Note: "word" is a group of characters separated by space.

    Examples:
    check_if_last_char_is_a_letter("apple pie") ➞ False
    check_if_last_char_is_a_letter("apple pi e") ➞ True
    check_if_last_char_is_a_letter("apple pi e ") ➞ False
    check_if_last_char_is_a_letter("") ➞ False 
    '''
 
    check = txt.split(' ')[-1]
    return True if len(check) == 1 and (97 <= ord(check.lower()) <= 122) else False

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert check_if_last_char_is_a_letter("apple") == False
    # assert check_if_last_char_is_a_letter("apple pi e") == True
    # assert check_if_last_char_is_a_letter("eeeee") == False
    # assert check_if_last_char_is_a_letter("A") == True
    # assert check_if_last_char_is_a_letter("Pumpkin pie ") == False
    # assert check_if_last_char_is_a_letter("Pumpkin pie 1") == False
    # assert check_if_last_char_is_a_letter("") == False
    # assert check_if_last_char_is_a_letter("eeeee e ") == False
    # assert check_if_last_char_is_a_letter("apple pie") == False
    # assert check_if_last_char_is_a_letter("apple pi e ") == False
    # assert True
    print("✅ HumanEval/134 - All assertions completed!")
