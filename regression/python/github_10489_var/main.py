def main() -> None:
    arr = [3, 5]
    ends = {}
    for i in range(2):
        if i == 1:
            k = ends[1]
            assert arr[k] < arr[i]
        ends[1] = i


main()
