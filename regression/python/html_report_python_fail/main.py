def double(n: int) -> int:
    return n * 2


x: int = double(3)
assert x > 10
# Keep a line below the violation: html.cpp renders a trace step at source
# line L only when the file has more than L lines (off-by-one at html.cpp:649).
y: int = x + 1
