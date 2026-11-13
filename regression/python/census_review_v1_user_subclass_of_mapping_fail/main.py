from collections.abc import Mapping


class Q(Mapping):

    def __getitem__(self, k):
        return k

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


print(list(Q.values([5, 6])))
