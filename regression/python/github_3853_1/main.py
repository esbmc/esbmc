class test1:
    def peak(self):
        pass

class test2(test1):
    def peak(self):
        pass

def test(a):
    a.peak()

test(test2())