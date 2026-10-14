# A heterogeneous int/float list read with a loop-variable (non-constant)
# index must read each element with its actual stored type. Reading the float
# element 2.0 as if it were an int (the type of element 0) produced a false
# counterexample (esbmc/esbmc#5160).
def tri(n):
    my_tri = [1, 3, 2.0]
    for i in range(3, n + 1):
        my_tri.append(my_tri[i - 1] + my_tri[i - 2] + (i + 3) / 2)
    return my_tri


if __name__ == "__main__":
    assert tri(3) == [1, 3, 2.0, 8.0]
