class MyClass:
    # Class attribute
    class_attr: int = 1
    mutable_attr: list = []  # Shared mutable attribute

    def __init__(self, value: int):
        self.data: int = value


class ChildClass(MyClass):
    child_attr: int = 100

    def __init__(self, value: int, child_value: int = 50):
        super().__init__(value)
        self.child_data = child_value


class GrandChildClass(ChildClass):
    class_attr: int = 999  # Override parent class attribute

    def __init__(self, value: int):
        super().__init__(value, 75)


# Checking if the class attribute has the expected value
assert MyClass.class_attr == 1
print("✓ Class attribute initial value correct")

# Creating an instance of MyClass
obj1 = MyClass(10)

# Instance attributes can be set with self.name = value
obj1.class_attr = 2

# Verifying that the instance attribute now hides the class attribute with the same name
assert obj1.class_attr == 2
print("✓ Instance attribute shadows class attribute")

# Confirming that the class attribute still retains its original value
assert MyClass.class_attr == 1
print("✓ Class attribute unchanged after instance attribute creation")

# Creating another instance of MyClass
obj2 = MyClass(15)

# Verifying that obj2 refers to the class attribute as it doesn't have an instance attribute with the same name
assert obj2.class_attr == 1
print("✓ New instance accesses class attribute when no instance attribute exists")

# Updating the class attribute, affecting all instances of MyClass
MyClass.class_attr = 3

# Checking that the class attribute has been successfully updated
assert MyClass.class_attr == 3
print("✓ Class attribute updated successfully")

# Verifying that the instance attribute of obj1 remains the same
assert obj1.class_attr == 2
print("✓ Instance attribute unaffected by class attribute change")

# Checking that obj2 now refers to the updated class attribute
assert obj2.class_attr == 3
print("✓ Instance without instance attribute sees updated class attribute")

# Test attribute deletion and fallback behavior
obj3 = MyClass(20)
obj3.class_attr = 99
assert obj3.class_attr == 99
print("✓ Instance attribute set")

# Delete instance attribute - should fall back to class attribute
del obj3.class_attr
assert obj3.class_attr == 3  # Falls back to class attribute
print("✓ After deleting instance attribute, falls back to class attribute")

# Test shared mutable attributes (common gotcha)
obj4 = MyClass(30)
obj5 = MyClass(40)

# Both instances share the same mutable class attribute
obj4.mutable_attr.append("from_obj4")
assert "from_obj4" in obj5.mutable_attr
print("✓ Mutable class attributes are shared between instances")

# Setting instance attribute doesn't affect class attribute
obj4.mutable_attr = ["instance_specific"]
assert obj4.mutable_attr == ["instance_specific"]
assert "from_obj4" in obj5.mutable_attr  # obj5 still sees class attribute
assert "from_obj4" in MyClass.mutable_attr  # Class attribute unchanged
print("✓ Setting instance attribute doesn't affect class mutable attribute")

# Test inheritance behavior
child1 = ChildClass(100)
assert child1.class_attr == 3  # Inherits from parent
assert child1.child_attr == 100
print("✓ Child class inherits parent class attributes")

# Test modifying parent class attribute affects children (unless overridden)
MyClass.class_attr = 555
child2 = ChildClass(300)
assert child2.class_attr == 555  # Gets updated parent value
print("✓ Parent class attribute changes affect children unless overridden")
