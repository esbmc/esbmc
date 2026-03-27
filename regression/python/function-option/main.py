def foo() -> None:
    pass

def bar() -> None:
    foo()
    x = int(5)
    bytes_data = b'\x00\x10'
    y = int.from_bytes(bytes_data, 'big', False)
    z = x.bit_length();
