# Negative twin of github_6273_receiver_call_once: the receiver call runs
# exactly once, so the dict holds one key and the two-key expectation fails.
a = {}
a.setdefault(len(a), []).append(1)
assert len(a) == 2
