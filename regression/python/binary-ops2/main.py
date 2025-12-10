s = ""
for _ in range(3):
    s = s + "a" + ("b" + "c")

assert s == "abcabcabc"
