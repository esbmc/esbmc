# Soundness guard for GitHub #4807. `classify` returns 'open' for this input,
# so `!= 'open'` is False and the assertion MUST fail (VERIFICATION FAILED).
#
# Before the fix, the nested bool-returning `has_open` leaked into classify's
# inferred return type, so classify was typed bool. The str-vs-bool `!=` then
# folded to a *constant True* (cross-type fold), turning a genuinely false
# assertion into VERIFICATION SUCCESSFUL — i.e. the bug masked a real bug.
# This test would have reported SUCCESSFUL under the old behaviour and so pins
# the soundness regression.


def classify(s):
    def has_open(t):
        for c in t:
            if c == '(':
                return True
        return False

    return 'open' if has_open(s) else 'none'


assert classify('(x)') != 'open'
