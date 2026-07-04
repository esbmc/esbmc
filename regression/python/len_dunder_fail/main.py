# __len__ returns 4, so len(c) is 4, not 5.
class C:
    def __len__(self):
        return 4


c = C()
assert len(c) == 5
