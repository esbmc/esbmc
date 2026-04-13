# example_12_string.py
class ExampleClass:
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return f"ExampleClass(name={self.name})"

def demonstrate_string_functions():
    # Basic string conversion
    obj = ExampleClass("test")
    print("str() of object:", str(obj))
    assert(str(obj) == "ExampleClass(name=test)")

    # # F-string formatting
    # class_name = "Worker"
    # variable_name = "count"
    # formatted = f"{class_name}_{variable_name}"
    # print("F-string formatted:", formatted)

    # # Different string formatting methods
    # count = 42
    # name = "Task"

    # # Using f-strings
    # print(f"There are {count} instances of {name}")

    # # Using .format()
    # print("There are {} instances of {}".format(count, name))

    # # Using % operator (older style)
    # print("There are %d instances of %s" % (count, name))

    # String methods
    # text = "  Hello, World!  "
    # print("Original:", text)
    # print("Stripped:", text.strip())
    # print("Upper:", text.upper())
    # print("Lower:", text.lower())
    # print("Replaced:", text.replace("Hello", "Hi"))

if __name__ == "__main__":
    demonstrate_string_functions()
