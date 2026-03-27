def test_chr_ord() -> bool:
    char_code = ord('B')  # Should be 66
    back_to_char = chr(char_code)
    return back_to_char == 'A' and char_code == 65

assert test_chr_ord() == True
