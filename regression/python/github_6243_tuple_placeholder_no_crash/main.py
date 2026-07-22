# Regression guard for #6243's fix: an earlier version retyped ANY existing
# symbol (denylisting only class-to-class) when rebound to a constructor
# call. A struct-shaped placeholder (here, a tuple) that already had a member
# read against it (`x = t[0]`) is NOT a safe retype target — the read is an
# expression built against the old struct layout, and retyping the symbol in
# place without fixing that expression up aborted GOTO conversion
# (member2t assertion). The fix now only widens from None/Any/scalar
# placeholders, leaving this case unretyped (and so out of #6243's scope);
# this test only pins that it no longer crashes.
class Service:
    def __init__(self, name):
        self._name = name


t = (1, 2)
x = t[0]
t = Service("hi")
assert t._name == "hi"
