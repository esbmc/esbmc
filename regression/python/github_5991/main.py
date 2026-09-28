# Regression for #5991: indexing a list that was appended-to inside a called
# function must not raise a spurious IndexError. The frontend's convert-time
# constant-index bounds check used the caller's static list length, which is
# blind to a mutation performed through a function argument; it now falls back
# to the runtime bounds check for lists that escape into a call.
def my_push(lst: list[int], item: int) -> None:
    lst.append(item)


h: list[int] = [2, 5, 8]
my_push(h, 1)

assert len(h) == 4
assert h[0] == 2
assert h[3] == 1  # the appended element — previously a spurious IndexError
