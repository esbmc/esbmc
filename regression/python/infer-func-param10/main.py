def get_value(data, key):
    return data.get(key, "not found")


assert get_value({"name": "Alice", "age": 30}, "name") == "Alice"
