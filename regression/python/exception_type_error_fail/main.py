try:
    if not isinstance("abc", int):
      raise TypeError("Not an int")
except TypeError as e:
    assert 0
