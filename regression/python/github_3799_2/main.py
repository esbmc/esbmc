# Deeper class hierarchy: Dog and Cat both inherit from Mammal(Animal).
# Common ancestor of Dog and Cat is Mammal (not Animal), so ESBMC
# should annotate 'animal' parameter as Mammal and call Mammal.speak().

class Animal:
    def speak(self):
        return -1

class Mammal(Animal):
    def speak(self):
        return 0

class Dog(Mammal):
    def speak(self):
        return 1

class Cat(Mammal):
    def speak(self):
        return 2

def make_sound(animal):
    result = animal.speak()
    assert result >= 0

d = Dog()
c = Cat()

make_sound(d)
make_sound(c)
