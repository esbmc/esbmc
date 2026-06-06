# Same "assert it raised" idiom as lower_exceptions_raise_assert, but the raised
# ValueError is not caught by the TypeError handler, so it escapes uncaught and
# the lowering's uncaught-exception assertion fires at __ESBMC_main -> FAILED.
try:
    raise ValueError("boom")
    assert False  # unreachable: the raise always fires
except TypeError:
    pass
