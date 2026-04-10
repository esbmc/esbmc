def read_value(d: dict, key):
    return d.get(key)

if __name__ == "__main__":
    data = {"a": 1, "b": 2}
    print(read_value(data, "a"))

    x = nondet_int()
    print(x + 1)
