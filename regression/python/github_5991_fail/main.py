# Negative counterpart to github_5991: suppressing the convert-time bounds
# check for call-escaped lists must not lose bounds checking entirely — a
# genuinely out-of-bounds access after the call is still caught, now via the
# runtime __ESBMC_list_at bound.
def my_push(lst: list[int], item: int) -> None:
    lst.append(item)


h: list[int] = [2, 5, 8]
my_push(h, 1)

# len(h) == 4, so index 4 is out of range and must fail.
assert h[4] == 0
