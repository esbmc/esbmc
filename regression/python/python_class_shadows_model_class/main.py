# KNOWNBUG. A class type symbol is keyed `tag-<Name>` with no module
# component, and an always-loaded operational model registers its classes under
# the program file, so a user class named after one of them is silently
# discarded: every use binds to the model's class and the assertion below is
# reported FAILED even though CPython holds it. The same shape with an
# *imported* model is refused instead (github_7397_model_shadow); only the
# always-loaded ones reach this path. Resolving it needs per-module class tags.
class Exception:
    def __init__(self) -> None:
        self.v: int = 2

    def get(self) -> int:
        return self.v


o = Exception()
assert o.get() == 2
