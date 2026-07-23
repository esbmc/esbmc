# KNOWNBUG, NOT closed by #6243's fix: the observable symptom here (a
# spurious out-of-bounds dereference inside Service.__init__) looks the same
# as the module-level rebind this issue fixes, but the root cause is
# different and untouched by that fix. A `global g` declaration inside a
# function routes `g = Service("hi")` through a separate code path that
# double-emits the constructor call (once against a throwaway `$ctor_self$`
# temp, once against `&g`) rather than a single, correctly-typed allocation.
# Confirmed pre-existing on unfixed master via bisection (this KNOWNBUG's
# failure is identical with and without #6243's fix applied) — it needs its
# own separate investigation into the global-variable write-back mechanism
# for function-scoped `global` declarations, not another retype-on-rebind
# fix.
class Service:

    def __init__(self, name):
        self._name = name
        self._tag = 1


g = 0


def make() -> None:
    global g
    g = Service("hi")


make()
