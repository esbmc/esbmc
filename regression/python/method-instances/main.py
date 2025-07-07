class MyClass:
    def __init__(self,x:int):
        self.x = x

    def return_x(self)->int:
        return self.x

myInstance = MyClass(5)

assert MyClass.return_x(self=myInstance) == 5

assert MyClass.return_x(myInstance) == 5