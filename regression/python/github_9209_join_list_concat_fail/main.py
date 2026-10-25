def g(b: list[str], c: list[str]) -> str:
    return ".".join(b + c)


assert g(["a"], ["b"]) == "b.a"
