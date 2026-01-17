# Test: String find - encontra substring
text = "hello world"
pos = text.find("world")
assert pos == 6
pos2 = text.find("hello")
assert pos2 == 0
pos3 = text.find("xyz")
assert pos3 == -1  # não encontrado
