# github.com/esbmc/esbmc/issues/6264 follow-up (#6273 regression guard)
# The list-method discriminator must not convert a receiver that contains a
# call: converting it re-emits the call, so setdefault runs twice and inserts a
# second key (the key varies with len(a)). One insertion is correct.
a = {}
a.setdefault(len(a), []).append(1)
assert len(a) == 1
