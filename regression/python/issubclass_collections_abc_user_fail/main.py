import collections.abc


class SeqLike:

    def __getitem__(self, i):
        return i

    def __len__(self):
        return 0


assert issubclass(SeqLike, collections.abc.Sequence)
