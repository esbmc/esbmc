import collections.abc


class Plain:
    pass


class WithIter:

    def __iter__(self):
        return self


class WithNext(WithIter):

    def __next__(self):
        return 1


class SeqLike:

    def __getitem__(self, i):
        return i

    def __len__(self):
        return 0


class WithEq:

    def __eq__(self, other):
        return True


class WithEqHash(WithEq):

    def __hash__(self):
        return 0


class MyList(list):
    pass


class Sequence:
    pass


assert not issubclass(Plain, collections.abc.Iterable)
assert issubclass(Plain, collections.abc.Hashable)
assert issubclass(WithIter, collections.abc.Iterable)
assert not issubclass(WithIter, collections.abc.Iterator)
assert issubclass(WithNext, collections.abc.Iterator)
assert not issubclass(SeqLike, collections.abc.Sequence)
assert not issubclass(SeqLike, collections.abc.Iterable)
assert issubclass(SeqLike, collections.abc.Sized)
assert not issubclass(WithEq, collections.abc.Hashable)
assert issubclass(WithEqHash, collections.abc.Hashable)
assert issubclass(MyList, collections.abc.MutableSequence)
assert not issubclass(MyList, collections.abc.Hashable)
assert not issubclass(list, Sequence)
