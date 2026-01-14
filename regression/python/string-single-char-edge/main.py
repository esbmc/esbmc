# Test: String com apenas um caractere
single = "a"
assert len(single) == 1
assert single[0] == "a"
assert single[-1] == "a"
assert single == "a"
assert single * 3 == "aaa"
assert single + "b" == "ab"
