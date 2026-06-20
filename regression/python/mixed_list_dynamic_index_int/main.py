# Reading the *integer* elements of a heterogeneous int/float list with a
# loop-variable index must promote them to float (Python int->float promotion),
# not read float_buf at the wrong slot. Covers the int branch of the runtime
# type dispatch added for esbmc/esbmc#5160.
def total(n):
    my_tri = [1, 3, 2.0]
    s = 0.0
    for i in range(0, n):
        s = s + my_tri[i]
    return s


if __name__ == "__main__":
    assert total(3) == 6.0
