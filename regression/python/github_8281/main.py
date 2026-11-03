import _sv_verifier as sv
import mod
import typing
from _sv_verifier import check_type as ct


class C:

    class D:
        pass


n = sv.nondet_int()
sv.check_type(n, int)
ct(n, int | str)
ct(None, None)
ct(None, type(None))
ct(None, int | None)
ct(C(), C)
ct(C.D(), C.D)
ct(mod.A(), mod.A)
ct(n, typing.Any)
ct([[n], [2]], list[list[int]])
ct({"a": 1, "b": 2}, dict[str, int])
ct((1, n), tuple[int, ...])
ct((n, "b"), tuple[int, str])
ct({n}, set[int])
