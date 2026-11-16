def mk():
    a = []
    a.append([1, 2])
    a.append([3])
    return a


assert mk() == [[1, 2], [4]]
