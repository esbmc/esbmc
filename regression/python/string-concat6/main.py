result = []
word = ""
for char in "a,":
    if char == ",":
        result.append(word)
    else:
        word += char
assert result[0] == "a"
