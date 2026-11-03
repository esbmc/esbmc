# Issue #9388: CPython calls g() before split("") raises, so the frontend
# refuses rather than drop the call.
def g() -> str:
    return "a b"


parts: list[str] = g().split("")
