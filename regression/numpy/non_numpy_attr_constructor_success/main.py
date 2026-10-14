class Factory:
    def zeros(self):
        return [1, 2, 3]


factory = Factory()
values = factory.zeros()
head = values[0:1]
values[0] = 9

assert head[0] == 1
assert values[0] == 9
