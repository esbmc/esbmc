class Factory:
    def eye(self, n):
        return [1, 2, 3]

factory = Factory()
n = factory.eye(3).size
