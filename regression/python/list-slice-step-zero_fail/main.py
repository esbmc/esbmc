# Regression for issue #4293: list slice with step==0.
# CPython raises ValueError("slice step cannot be zero"); ESBMC raises a
# catchable ValueError, and this uncaught one makes verification fail here.
xs = [10, 20, 30]
ys = xs[::0]
