# Regression for #5159: a string element unpacked from str.split() and rebound
# to the SAME name via int() must not be truncated. int() now yields a 64-bit
# Python int, so the string pointer survives the assignment and is parsed
# correctly (a 32-bit result truncated the pointer, giving a wrong value and an
# out-of-bounds read in the `month in [...]` membership test).


def first_field_in_months(date: str) -> bool:
    month, day = date.split('-')
    month = int(month)
    return month in [1, 3, 5, 7, 8, 10, 12]


if __name__ == "__main__":
    assert first_field_in_months('3-11') == True
    assert first_field_in_months('4-11') == False
