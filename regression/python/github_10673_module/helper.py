def g(x=1):
    return x


h = g
r = h(x=2)

if __name__ == "__main__":
    print(r)
