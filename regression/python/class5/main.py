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

person1 = Person(101, 25)
assert person1.is_adult()
person2 = Person(202, 18)
assert person2.is_adult()

assert person1.greet() == "Hello"
assert person2.greet() == "Hello"

assert person1.have_birthday() == "Happy birthday!"
assert person1.get_old() == 26

assert person2.have_birthday() == "Happy birthday!"
assert person2.get_old() == 19
