# pop() on the empty list a plain function returns raises IndexError (#10827).
def send() -> list[list[str]]:
    return []


def f() -> None:
    send().pop()


f()
