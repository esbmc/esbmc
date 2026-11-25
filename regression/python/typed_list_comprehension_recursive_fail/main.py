def quicksort(arr):
    if not arr:
        return []
    pivot = arr[0]
    lesser = quicksort([x for x in arr[1:] if x < pivot])
    greater = quicksort([x for x in arr[1:] if x > pivot])
    return lesser + [pivot] + greater


def main() -> None:
    l = [3, 1, 2, 3]
    assert quicksort(l) == sorted(l)


main()
