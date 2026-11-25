def f(arr: list[int]) -> int:
    ends = {}
    n = 0
    for i, v in enumerate(arr):
        if i > 0 and arr[ends[1] + 5] < v:
            n = n + 1
        ends[1] = i
    return n


def main() -> None:
    arr = [3, 7]
    for i in range(len(arr)):
        assert arr[i] > 0
    assert f(arr) == 1


main()
