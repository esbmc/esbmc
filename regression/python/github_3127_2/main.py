def my_split(s: str, sep: str) -> list[str]:
    result = []
    word = ""
    for char in s:
        if char == sep:
            result.append(word)
            word = ""
        else:
            word += char
    result.append(word)
    return result

s: str = "a,b"
l = my_split(s, ",")
assert len(l) == 2
assert l[0] == "a"
assert l[1] == "b"
assert l == ["a", "b"]
