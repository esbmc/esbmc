
import string
def to_base(num, b):
    result = ''
    alphabet = string.digits + string.ascii_uppercase
    while num > 0:
        i = num % b
        num = num // b
        result = alphabet[i] + result
    return result

"""
import string
def to_base(num, b):
    result = ''
    alphabet = string.digits + string.ascii_uppercase
    while num > 0:
        i = num % b
        num = num // b
        result = result + alphabet[i]
    return result[::-1]
"""

assert to_base(8227, 18) == "1771"
assert to_base(73, 8) == "111"
assert to_base(16, 19) == "G"
assert to_base(31, 16) == "1F"
assert to_base(41, 2) == "10100" # this should fail
assert to_base(44, 5) == "134"
assert to_base(27, 23) == "14"
assert to_base(56, 23) == "2A"
assert to_base(8237, 24) == "E75"
assert to_base(8237, 34) == "749"
