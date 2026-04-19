class MyClass:
    class_attr: int = 1
    mutable_attr: list = []
    
    def __init__(self, value: int):
        self.data: int = value

def test_mutable_class_attributes():
    # Reset mutable_attr for clean test
    MyClass.mutable_attr = []
    
    obj1 = MyClass(30)
    obj2 = MyClass(40)
    
    # Both instances share the same mutable class attribute
    obj1.mutable_attr.append("from_obj1")
    assert "from_obj1" in obj2.mutable_attr
    
    # Setting instance attribute doesn't affect class attribute
    obj1.mutable_attr = ["instance_specific"]
    assert obj1.mutable_attr == ["instance_specific"]
    assert "from_obj1" in MyClass.mutable_attr
    print("✓ Mutable class attributes behave correctly")

test_mutable_class_attributes()
