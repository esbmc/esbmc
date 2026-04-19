# Issue #3855: float variable reassigned to chr() result (dynamic type change).
# The assignment is not modelled, but the chr() call must still execute so that
# out-of-range arguments raise ValueError.  chr(65) is in range, so no exception.
a = 60
b = 5
sum = float(a + b)
assert sum == 65.0
sum = chr(int(sum))  # type change float->str not modelled; void call preserves exceptions
