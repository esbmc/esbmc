def return_sum(x:int=0,y:int=0) -> int:
    return x/y

x:int = return_sum(y=1,x=2)
assert x==2