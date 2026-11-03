def via_name(s: str) -> str:
    parts = s.split(" ")
    return " ".join(parts)


def via_call(s: str) -> str:
    return "-".join(s.split(" "))


assert via_name("a b") == "a b"
assert via_call("a b") == "a b"
