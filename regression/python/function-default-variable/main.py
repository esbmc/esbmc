x:int=1

def return_int(y:int=x)->int:
    return y

assert return_int() == 1

x = 2

assert return_int() == 1