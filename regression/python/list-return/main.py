def return_list(myList: list[int]) -> list[int]:
    tempList = myList
    return tempList


list1 = [1, 2, 3]
list2 = return_list(list1)

assert list2[0] == 1
