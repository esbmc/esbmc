class Dog:
    def speak(self) -> str:
        return "Woof!"

def test():
    d = Dog()
    x: float = d.speak()
    assert x == "Woof!"

test()
