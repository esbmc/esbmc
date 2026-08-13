# A method reading self.<attr> on a constructor temporary: the receiver must
# run __init__ before compute() reads self.x.
class C:
    def __init__(self, v):
        self.x = v

    def compute(self):
        return self.x * 2


assert C(5).compute() == 10
