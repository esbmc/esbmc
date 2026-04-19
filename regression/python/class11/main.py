class MyClass:
    class_attr: int = 1
    mutable_attr: list = []
    
    def __init__(self, value: int):
        self.data: int = value

def test_basic_class_instance_attributes():
    assert MyClass.class_attr == 1
    
    obj1 = MyClass(10)
    obj1.class_attr = 2
    assert obj1.class_attr == 2
    assert MyClass.class_attr == 1
    
    obj2 = MyClass(15)
    assert obj2.class_attr == 1
    
    MyClass.class_attr = 3
    assert MyClass.class_attr == 3
    assert obj1.class_attr == 2
    assert obj2.class_attr == 3
    
test_basic_class_instance_attributes()

