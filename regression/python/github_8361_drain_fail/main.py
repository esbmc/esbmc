def count(words: list[str]) -> int:
    n = 0
    while words and (w := words.pop()):
        n += 1
    return n


assert count(["a", "b"]) == 3
