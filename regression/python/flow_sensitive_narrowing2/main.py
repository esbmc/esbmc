def normalize(x):
    result = []
    i = 0
    types = [0, 0, 0, 0]
    
    

    while i < 4:
         if isinstance(x, int):
              x = str(x)
              types[i] = 1

         elif isinstance(x, str):
              x = [x]
              types[i] = 2

         elif isinstance(x, list):
              x = len(x)
              types[i] = 3

         else:
              x = 0
              types[i] = 4


         result.append(x)
         i += 1

    assert len(result) == 4
    assert len(types) == 4

#    assert types[0] == 1
#    assert types[1] == 2
#    assert types[2] == 3 
#    assert types[3] == 1
    assert isinstance(result[0], str)
    assert isinstance(result[1], list) 
    assert isinstance(result[2], int)
    assert isinstance(result[3], str)

    return result


normalize(5)

