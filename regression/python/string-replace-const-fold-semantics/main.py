s = "hello world"

assert s.replace("o", "0") == "hell0 w0rld"
assert s.replace("l", "L", 2) == "heLLo world"
assert s.replace("zz", "q") == "hello world"
assert s.replace("hello world", "") == ""

overlapping = "aaa"
assert overlapping.replace("aa", "b") == "ba"
