def normalize(x):
    result = []
    i = 0

    while i < 4:
         if isinstance(x, int):
              x = str(x)
         elif isinstance(x, str):
              x = [x]
         elif isinstance(x, list):
              x = len(x)
         else:
              x = 0


         result.append(x)
         i += 1

    assert len(result) == 4


    assert isinstance(result[0], str)
    assert isinstance(result[1], list)
    assert isinstance(result[2], int) 
    assert isinstance(result[3], str)


    return result


normalize(5)

