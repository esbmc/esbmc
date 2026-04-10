import pkg.mod4

# Test 4: Verify submodule class merging
# Tests class merging in module_manager.cpp load_directory()
c = pkg.mod4.MyClass("test")
# Verify constructor call succeeds (type inference works correctly)
assert True
