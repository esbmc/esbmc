class Foo:
    def __init__(self, s: str) -> None:
        t = s.lower()
        self.x: str = t

def test_lower():
    # Test basic uppercase conversion
    foo1 = Foo("HELLO")
    assert foo1.x[0] == 'h'
    assert foo1.x[1] == 'e'
    assert foo1.x[4] == 'o'
    
    # Test mixed case conversion
    foo2 = Foo("HeLLo")
    assert foo2.x[0] == 'h'
    assert foo2.x[2] == 'l'
    
    # Test already lowercase (should remain unchanged)
    foo3 = Foo("hello")
    assert foo3.x[0] == 'h'
    assert foo3.x[4] == 'o'
    
    # Test with numbers and special chars (should remain unchanged)
    foo4 = Foo("Hello123")
    assert foo4.x[0] == 'h'
    assert foo4.x[5] == '1'

test_lower()
