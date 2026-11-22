def bucketsort(arr: list[int], k: int) -> list[int]:
    counts = [0] * k
    for x in arr:
        counts[x] += 1
    sorted_arr = []
    for i, count in enumerate(counts):
        sorted_arr.extend([i] * count)
    return sorted_arr


def main() -> None:
    arr = nondet_list(nondet_int)
    k = nondet_int()
    if not (len(arr) <= 2 and 0 <= k <= 2):
        return
    for x in arr:
        if not (0 <= x < k):
            return
    assert bucketsort(arr, k) == sorted(arr)


main()
