def first_positive(values):
    for i in range(len(values)):
        if values[i] > 0:
            return list(values)
    return None


assert first_positive([1]) == [2]
