x = 10

if isinstance(x, int):
    print("x is an int")

key_type = str  # key_type stores a type

if key_type is str:
    print("key_type represents the 'str' type")

y = 5

if y is int:
    print("This will never print")
else:
    print("y is int -> False (wrong way to check)")
