# Negative companion to int_from_split_rebind (#5159). The rebound int value is
# parsed correctly, so the membership test is genuinely False here and the
# assertion that it is True must make verification FAIL — pinning that int() of
# a split-unpacked, self-rebound variable produces a real value (not nondet or a
# vacuously-true claim).


def first_field_in_months(date: str) -> bool:
    month, day = date.split('-')
    month = int(month)
    return month in [1, 3, 5, 7, 8, 10, 12]


if __name__ == "__main__":
    # 4 is not in the odd-month set, so this is False; asserting True must FAIL.
    assert first_field_in_months('4-11') == True
