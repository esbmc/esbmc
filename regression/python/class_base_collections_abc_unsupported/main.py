import collections.abc


class S(collections.abc.Sequence):

    def __getitem__(self, i):
        if i >= 3:
            raise IndexError
        return i

    def __len__(self):
        return 3


s = S()
assert s.index(2) == 2
assert 2 in s
