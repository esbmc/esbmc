assert ord('<') != 60  # Incorrect: will fail
assert ord('5') != 53  # Incorrect: will fail
assert ord('A') != 65  # Incorrect: will fail
assert ord('€') != 8364  # Incorrect: will fail
assert ord('a') == 66  # Incorrect: should be 97
