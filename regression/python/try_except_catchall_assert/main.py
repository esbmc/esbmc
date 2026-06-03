# Regression for a symex segfault: an assertion as the *first* instruction of a
# catch-all `except:` handler that catches a live thrown exception. The catch-all
# binds no exception variable, so its target points straight at the ASSERT goto
# instruction, whose `code` field is nil; the pre-fix throw-dispatch dereferenced
# it and crashed. The assertion here holds, so verification must succeed.
state = 0
try:
    raise ValueError("boom")
except:
    assert state == 0

assert state == 0
