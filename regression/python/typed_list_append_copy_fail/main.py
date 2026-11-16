def copy(arr: list[int]) -> list[int]:
    b = []
    for x in arr:
        b.append(x + 1)
    return b


def main() -> None:
    arr = nondet_list(nondet_int)
    if len(arr) > 3:
        return
    b = copy(arr)
    i = 0
    while i < len(arr):
        assert b[i] == arr[i]
        i += 1


main()
