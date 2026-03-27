class Foo:
    def __init__(self) -> None:
        pass
    
    def foo(self, *, l: list[str]):
        assert len(l) > 0
        
        count = 0
        for i in l:
            assert isinstance(i, str)
            assert i is None
            count += 1
        
        # Assert we iterated through all elements
        assert count == len(l)

f = Foo()
test_list = ["hello", "world", "test"]
f.foo(l=test_list)
f.foo(l=["single"])
