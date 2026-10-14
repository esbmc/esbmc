# Comparing an inner list read out of a list parameter with another list:
# list_eq used to read the pair back from its stack arrays, which lost the
# inner list's object, so each element width was unknown and the byte
# compare never stopped unwinding.
def check(inp: list[list[int]], exp: list[int]) -> None:
    assert inp[0] == exp


check([[1, 4, 1]], [1, 4, 2])
