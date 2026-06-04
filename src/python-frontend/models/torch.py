# pylint: disable=undefined-variable,unused-argument,redefined-builtin
# Operational model for a subset of PyTorch (torch).
#
# These pure-Python reference implementations let ESBMC verify programs that use
# the modelled torch operations. Tensors are modelled as nested Python lists of
# floats (the default dtype of torch.randn and the common ML case). Element
# types are annotated as float because the converter does not yet propagate a
# caller's element type into a model function: without the annotation, indexed
# tensor elements are `any`-typed and arithmetic over them is over-approximated
# as nondet, which trips the type system. (Delegating to numpy.matmul does not
# help — numpy's matmul lowering requires literal-shaped arguments at the call
# site and so fails through any wrapper.) Argument names are part of the
# contract matched by the converter.
#
# Modelled: mm, matmul, cat, split, allclose, tensor.
import math


def tensor(data: list[list[float]]) -> list[list[float]]:
    # Tensors are modelled directly as nested Python lists; tensor() is identity.
    return data


def mm(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    # 2-D matrix multiply: A is (n x k), B is (k x m) -> (n x m).
    n = len(A)
    k = len(B)
    m = len(B[0])
    C = [[0.0 for _ in range(m)] for _ in range(n)]
    i = 0
    while i < n:
        j = 0
        while j < m:
            s = 0.0
            t = 0
            while t < k:
                s = s + A[i][t] * B[t][j]
                t = t + 1
            C[i][j] = s
            j = j + 1
        i = i + 1
    return C


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    # 2-D alias of mm (the n-D broadcasting cases are not modelled yet).
    return mm(A, B)


def cat(tensors: list[list[list[float]]], dim: int) -> list[list[float]]:
    # Concatenate a list of 2-D tensors along columns (dim == 1): the rows stay
    # aligned and the columns of each tensor are appended. (dim == 0 row-stacking
    # is not modelled here.) Uses pre-sized index assignment rather than repeated
    # list concatenation, which would generate expensive per-element memcpy loops.
    n = len(tensors[0])
    total = 0
    ti = 0
    while ti < len(tensors):
        total = total + len(tensors[ti][0])
        ti = ti + 1
    out = [[0.0 for _ in range(total)] for _ in range(n)]
    r = 0
    while r < n:
        col = 0
        ti = 0
        while ti < len(tensors):
            c = 0
            while c < len(tensors[ti][r]):
                out[r][col] = tensors[ti][r][c]
                col = col + 1
                c = c + 1
            ti = ti + 1
        r = r + 1
    return out


def split(
    tensor: list[list[float]], sizes: list[int], dim: int
) -> list[list[list[float]]]:
    # Split a 2-D tensor along columns (dim == 1) into chunks of the given widths.
    parts = []
    start = 0
    si = 0
    while si < len(sizes):
        width = sizes[si]
        n = len(tensor)
        chunk = [[0.0 for _ in range(width)] for _ in range(n)]
        r = 0
        while r < n:
            c = 0
            while c < width:
                chunk[r][c] = tensor[r][start + c]
                c = c + 1
            r = r + 1
        parts = parts + [chunk]
        start = start + width
        si = si + 1
    return parts


def allclose(
    a: list[list[float]],
    b: list[list[float]],
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> bool:
    # Element-wise |a - c| <= atol + rtol*|c| over two 2-D tensors of equal shape.
    i = 0
    while i < len(a):
        j = 0
        while j < len(a[i]):
            if math.fabs(a[i][j] - b[i][j]) > atol + rtol * math.fabs(b[i][j]):
                return False
            j = j + 1
        i = i + 1
    return True
