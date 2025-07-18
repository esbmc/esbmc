class MyClass:
   attr0:int = 1

# Modify a class attribute in a different scope
def mutate_object(obj: MyClass) -> MyClass:
    obj.attr0 = 5
    return obj

myInstance:MyClass = MyClass()

returned_obj = mutate_object(myInstance)

assert returned_obj.attr0 == 5
assert myInstance.attr0 == 5
assert MyClass.attr0 == 1