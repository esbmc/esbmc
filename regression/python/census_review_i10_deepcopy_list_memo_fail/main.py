import copy


class M(list):

    def get(self, k, d=None):
        return d


print(copy.deepcopy([1], M()))
