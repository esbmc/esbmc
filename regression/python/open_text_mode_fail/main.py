try:
    f = open("missing.txt")
    f.write("x")
except ValueError:
    pass
