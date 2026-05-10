def main() -> None:
    pairs = [(1, "b"), (3, "a"), (2, "c")]

    # sorted(key=) on a list-of-tuples literal.
    out = sorted(pairs, key=lambda p: p[1])
    assert out[0][1] == "a"

    # list.sort(key=) — mutate in place; rewrites to `pairs = sorted(...)`.
    pairs.sort(key=lambda p: p[1])
    assert pairs[0][1] == "a"
    assert pairs[2][1] == "c"

    # Numeric sort key.
    nums = [(5, "x"), (1, "y"), (3, "z")]
    nums.sort(key=lambda p: p[0])
    assert nums[0][0] == 1
main()
