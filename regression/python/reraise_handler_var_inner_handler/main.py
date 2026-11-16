# `raise e` inside an inner handler re-raises the ValueError the outer handler
# bound to e, not the KeyError the inner handler bound to k.
flag = False
result = 0
try:
    try:
        raise ValueError()
    except Exception as e:
        try:
            raise KeyError()
        except KeyError as k:
            if flag:
                raise k
            raise e
except ValueError:
    result = 1
except KeyError:
    result = 2
assert result == 1
