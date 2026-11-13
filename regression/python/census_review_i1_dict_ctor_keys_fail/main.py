class T(list):

    def keys(self):
        return [0]


print(dict(T()))
