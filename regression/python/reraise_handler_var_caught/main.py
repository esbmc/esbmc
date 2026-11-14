# The re-raised IndexError is caught by an outer `except IndexError`.
l = [1, 2]
caught = False
try:
    try:
        x = l[5]
    except Exception as e:
        raise e
except IndexError:
    caught = True
assert caught
