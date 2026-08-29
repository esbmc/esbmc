# VSA aborted on every Python program -- the always-linked operational model
# lowers an assignment whose target is a conditional, which value_sett::assign_rec
# had no case for -- so --gcse and every other points-to consumer silently lost
# its data. Any program at all exercises it.
x: int = 1
assert x == 1
