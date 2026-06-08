# Regression for issue #4293: list slice with step==0.
# CPython raises ValueError("slice step cannot be zero"); ESBMC reports it
# via a failing assertion so verification flags the bad slice.
xs = [10, 20, 30]
ys = xs[::0]
