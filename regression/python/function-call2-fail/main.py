def foo(s: str) -> None:
    x: int = len(s)
    assert x == 4

def call_foo_direct():
    foo("test")  # Correct length: passes

def call_foo_indirect():
    s: str = "abcd"
    foo(s)  # Also correct: passes

def call_foo_fail():
    foo("fail!")  # Length is 5: should trigger assertion

def main():
    call_foo_direct()
    call_foo_indirect()
    call_foo_fail()

main()
