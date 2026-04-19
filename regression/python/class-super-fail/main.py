class Animal:
    def speak(self):
        assert(False)

class Dog(Animal):
    def speak(self):
        super().speak()  # Call the parent class's method

d = Dog()
d.speak()
