class MyClass:
    # Class attribute
    class_attr:int = 1

    def __init__(self, value: int):
        self.data:int = value


# Checking if the class attribute has the expected value
assert MyClass.class_attr == 1


# Creating an instance of MyClass
obj1 = MyClass(10)

# Instance attributes can be set with self.name = value
obj1.class_attr = 2

# Verifying that the instance attribute now hides the class attribute with the same name
assert obj1.class_attr == 2

# Confirming that the class attribute still retains its original value
assert MyClass.class_attr == 1

# Creating another instance of MyClass
obj2 = MyClass(15)

# Verifying that obj2 refers to the class attribute as it doesn't have an instance attribute with the same name
assert obj2.class_attr == 1


# Updating the class attribute, affecting all instances of MyClass
MyClass.class_attr = 3

# Checking that the class attribute has been successfully updated
assert MyClass.class_attr == 3

# Verifying that the instance attribute of obj1 remains the same
assert obj1.class_attr == 2

# Checking that obj2 now refers to the updated class attribute
assert obj2.class_attr == 3
