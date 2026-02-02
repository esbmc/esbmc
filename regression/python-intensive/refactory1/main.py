def remove_extras(lst):
    newlist = []
    for i in lst:
        if i not in newlist:
            newlist.append(i)
    return newlist

assert remove_extras([1, 1, 1, 2, 3]) == [1, 2, 3]
