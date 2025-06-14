def int_annotation() -> "int":
    return 42

def str_annotation() -> "str":
    return "hello"

def float_annotation() -> "float":
    return 3.14

def bool_annotation() -> "bool":
    return True

result1 = int_annotation()
assert result1 == 42
result2 = str_annotation()
assert result2 == "hello"
result3 = float_annotation()
assert result3 == 3.14
result4 = bool_annotation()
assert result4 == True
