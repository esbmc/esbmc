# github #9960: an int above 2**53 is not equal to the nearest float.
assert [9007199254740992] in [[9007199254740992.0]]
assert [9007199254740993] in [[9007199254740992.0]]
