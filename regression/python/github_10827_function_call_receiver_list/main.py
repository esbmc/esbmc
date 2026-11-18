# A list method on a plain function call receiver runs the call (#10827).
calls: int = 0


def send() -> list[int]:
    global calls
    calls += 1
    return [1, 2]


def f() -> None:
    assert send().pop() == 2
    send().append(3)
    assert calls == 2


f()
