# Test 1: Basic del and membership check
my_dict = {"a": 1, "b": 2, "c": 3}
assert (my_dict["a"] == 1)
del my_dict["b"]
assert "b" not in my_dict

# Test 2: Verify remaining keys still exist after del
my_dict2 = {"x": 10, "y": 20, "z": 30}
del my_dict2["y"]
assert "x" in my_dict2
assert "z" in my_dict2
assert my_dict2["x"] == 10
assert my_dict2["z"] == 30

# Test 3: Delete multiple keys sequentially
my_dict3 = {"a": 1, "b": 2, "c": 3, "d": 4}
del my_dict3["a"]
del my_dict3["c"]
assert "a" not in my_dict3
assert "b" in my_dict3
assert "c" not in my_dict3
assert "d" in my_dict3

# Test 4: Delete first key
my_dict4 = {"first": 100, "second": 200}
del my_dict4["first"]
assert "first" not in my_dict4
assert "second" in my_dict4

# Test 5: Delete last key
my_dict5 = {"first": 100, "second": 200}
del my_dict5["second"]
assert "first" in my_dict5
assert "second" not in my_dict5

# Test 6: Delete all keys
my_dict6 = {"only": 999}
del my_dict6["only"]
assert "only" not in my_dict6

# Test 7: Dict with different value types
my_dict7 = {"int_key": 42, "str_key": "hello", "float_key": 3.14}
del my_dict7["str_key"]
assert "str_key" not in my_dict7
assert my_dict7["int_key"] == 42

# Test 8: Verify "in" returns False after deletion
my_dict9 = {"key": 123}
assert "key" in my_dict9
del my_dict9["key"]
assert not ("key" in my_dict9)
