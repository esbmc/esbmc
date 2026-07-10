# Stale-seed false proof through an operand: x seeded as [1] made the literal
# receiver's index() fold to 0 (GitHub #5955), but CPython's x is [1, 2] at
# the call, so the assert must not verify. (Post-fix the unseeded x reaches
# the list model, whose nested-element matching raises ValueError here — a
# separate limitation; either way this is VERIFICATION FAILED, not a proof.)
x = [1]
x.append(2)
assert [[1], [1, 2]].index(x) == 0
