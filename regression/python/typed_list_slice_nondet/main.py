def main() -> None:
    arr = nondet_list(nondet_int)
    n = len(arr)
    if n > 4:
        return
    i = nondet_int()
    j = nondet_int()

    b = arr[i:j]
    s = i
    if s < 0:
        s += n
        if s < 0:
            s = 0
    elif s > n:
        s = n
    e = j
    if e < 0:
        e += n
        if e < 0:
            e = 0
    elif e > n:
        e = n
    assert len(b) == (e - s if e > s else 0)
    k = 0
    while k < len(b):
        assert b[k] == arr[s + k]
        k += 1

    c = arr[i:j:-1]
    s = i
    if s < 0:
        s += n
        if s < 0:
            s = -1
    elif s >= n:
        s = n - 1
    e = j
    if e < 0:
        e += n
        if e < 0:
            e = -1
    elif e >= n:
        e = n - 1
    assert len(c) == (s - e if s > e else 0)
    k = 0
    while k < len(c):
        assert c[k] == arr[s - k]
        k += 1


main()
