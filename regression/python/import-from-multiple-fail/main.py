from other import OtherClass, sum

obj = OtherClass()

assert obj.foo() == 3
assert sum(1,2) == 3
sub(3,2) # Invoking a function that hasn't been imported