class MyClass:
    value:int = 1


x:MyClass = MyClass()

def return_obj(y:MyClass=x)->MyClass:
    return y

obj:MyClass = return_obj()
assert obj.value == 1
assert x.value == 1

x = MyClass()
x.value = 2

obj:MyClass = return_obj()
assert obj.value==1
assert x.value == 2