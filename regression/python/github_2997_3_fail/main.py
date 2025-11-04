class Foo:  
    def __init__(self, x: int) -> None:  
        self.x = x  
  
    def bar(self) -> 'Bar':  
        return Bar(self)  
  
class Bar:  
    def __init__(self, f: Foo) -> None:  
        self.x: int = f.x  

f = Foo(1)
b = f.bar()
assert b.x == 2