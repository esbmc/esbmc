def foo(x: str) -> bool:
    return (int(x) <= 5) and True


assert foo("2")
