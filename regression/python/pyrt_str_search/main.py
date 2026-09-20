assert "Hello".upper() == "HELLO"
assert "Hello".lower() == "hello"
assert "hello world".title() == "Hello World"
assert "ab1cd".title() == "Ab1Cd"
assert "".upper() == ""

assert "abc".startswith("ab")
assert not "abc".startswith("bc")
assert "abc".endswith("bc")
assert not "abc".endswith("ab")
assert "ab".startswith("abcd") is False

assert "abcabc".find("c") == 2
assert "abcabc".rfind("c") == 5
assert "abcabc".find("z") == -1
assert "abcabc".index("b") == 1

assert "aaa".count("aa") == 1
assert "banana".count("a") == 3
assert "banana".count("z") == 0

assert "b" in "abc"
assert "z" not in "abc"
assert "bc" in "abc"
