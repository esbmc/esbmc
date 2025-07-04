class Person:
    def __init__(self, name: int, age: int):
        self.name = name
        self.age = age
    
    def greet(self) -> str:
        return "Hello"
    
    def have_birthday(self) -> str:
        return "Happy birthday!"
    
    def get_old(self) -> int:
      self.age += 1
      return self.age
 
    def is_adult(self) -> bool:
        return self.age >= 18

# Person just under adulthood
person3 = Person(303, 17)
assert not person3.is_adult()
assert person3.greet() == "Hello"
assert person3.have_birthday() == "Happy birthday!"
assert person3.get_old() == 18
assert person3.is_adult()  # Now should be adult

# Person with high ID and high age
person4 = Person(999, 99)
assert person4.is_adult()
assert person4.greet() == "Hello"
assert person4.have_birthday() == "Happy birthday!"
assert person4.get_old() == 100
assert person4.age == 100

# Person with zero age
person5 = Person(0, 0)
assert not person5.is_adult()
assert person5.greet() == "Hello"
assert person5.have_birthday() == "Happy birthday!"
assert person5.get_old() == 1
assert not person5.is_adult()

# Person turning exactly 18
person6 = Person(606, 17)
assert not person6.is_adult()
person6.get_old()  # Now 18
assert person6.is_adult()

