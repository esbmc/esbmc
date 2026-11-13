# A KeyError caught inside the handler does not change what `raise e`
# re-raises: it is still the ValueError bound to e.
result = 0
try:
    try:
        raise ValueError()
    except Exception as e:
        try:
            raise KeyError()
        except KeyError:
            pass
        raise e
except ValueError:
    result = 1
except KeyError:
    result = 2
assert result == 1
