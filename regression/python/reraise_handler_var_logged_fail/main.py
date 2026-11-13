# Reading e before the raise, as logging does, leaves e bound to the caught
# IndexError.
l = [1, 2]
try:
    x = l[5]
except Exception as e:
    msg = str(e)
    raise e
