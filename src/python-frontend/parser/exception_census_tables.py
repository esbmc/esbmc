"""Allowlist tables for exception_census.py.  Everything not listed here is rejected.

Each table was checked against tools/audit_stdlib.py output (sampled
arguments, CPython 3.14) and against the documentation.  A name is listed only
if it cannot raise the family on any argument, other than through a callable
the program itself supplies (which exception_census.py walks).
"""
# pylint: disable=missing-function-docstring,invalid-name
import builtins
import importlib
import sys

# The interpreter the tables were checked against, and the one the competition's
# Ubuntu 26.04 image ships: library behaviour differs between minor versions (for
# instance random.sample([], 0, counts=[])), so any other interpreter declines.
SUPPORTED_PYTHON = (3, 14)

# sha256 of the sv-benchmarks python/_sv_verifier.py that was read and
# brute-force tested (no nondet_* or check_type call raised either family).  A
# copy next to the program that differs makes the census decline; an absent
# copy is the audited module, as the cloud worker's task alone in a directory.
AUDITED_SV_VERIFIER = frozenset(
    {"41b7642036e03466637f77c922b976d1c5dbf741d0a3373ba92ce8298125db53"})

FAMILIES = {
    "arithmetic": ("ZeroDivisionError", "FloatingPointError"),
    "data-lookup": ("KeyError", "IndexError"),
}

# sys.stdlib_module_names exists from 3.10; the fallback is a short list of
# modules the allowlisted code can import transitively.
STDLIB_NAMES = frozenset(getattr(sys, "stdlib_module_names", ())) | frozenset(
    """abc array collections copy dataclasses enum functools heapq itertools math random
    string types typing sys os re io warnings operator keyword reprlib weakref contextlib
    bisect hashlib struct time datetime decimal fractions statistics numbers site""".split())

EXC_NAMES = frozenset(
    n for n in dir(builtins)
    if isinstance(getattr(builtins, n), type) and issubclass(getattr(builtins, n), BaseException))

_BUILTINS_COMMON = """abs all any ascii bin bool bytearray bytes callable chr classmethod
    complex dict enumerate filter float format frozenset hasattr hash hex id int isinstance
    issubclass iter len list map max min next object oct ord print property range repr
    reversed round set slice sorted staticmethod str sum super tuple zip
    Ellipsis NotImplemented __name__ __debug__""".split()

# divmod and pow raise ZeroDivisionError; neither raises a LookupError.
SAFE_BUILTINS = {
    "arithmetic": frozenset(_BUILTINS_COMMON) | EXC_NAMES,
    "data-lookup": frozenset(_BUILTINS_COMMON + ["divmod", "pow"]) | EXC_NAMES,
}

# Names that the program may call as intrinsics without defining them.
INTRINSICS = frozenset("nondet_int nondet_bool nondet_str nondet_float nondet_list nondet_dict "
                       "__ESBMC_assume".split())

# Dunder methods a program may define.  Every body is walked, so these are
# safe to define; the omitted ones (__getattr__, __new__, ...) change how
# attribute and construction dispatch resolves.
SAFE_DUNDER_DEFS = frozenset(
    """__init__ __repr__ __str__ __eq__ __ne__ __lt__ __le__ __gt__ __ge__ __hash__ __bool__
    __len__ __iter__ __next__ __contains__ __call__ __enter__ __exit__ __add__ __radd__ __iadd__
    __sub__ __rsub__ __isub__ __mul__ __rmul__ __imul__ __neg__ __pos__ __abs__ __invert__
    __and__ __rand__ __or__ __ror__ __xor__ __rxor__ __lshift__ __rshift__ __truediv__
    __rtruediv__ __floordiv__ __rfloordiv__ __mod__ __rmod__ __pow__ __rpow__ __divmod__
    __getitem__ __setitem__ __delitem__ __reversed__ __index__ __int__ __float__ __format__
    __matmul__ __rmatmul__ __post_init__""".split())

SAFE_DUNDER_STORES = SAFE_DUNDER_DEFS | {"__slots__", "__all__", "__match_args__"}

TYPE_GENERICS = frozenset(
    """list dict set frozenset tuple type List Dict Set FrozenSet Tuple Type Optional Union
    Callable Iterable Iterator Sequence Mapping MutableMapping MutableSequence MutableSet
    Generator Literal Annotated ClassVar Final Generic Protocol Deque DefaultDict Counter
    OrderedDict deque defaultdict Collection Container AbstractSet TypeGuard""".split())
TYPING_MODULES = frozenset({"typing", "collections", "collections.abc", "__future__"})

# Methods of the allowed builtin and library types.  `pop`-style names that
# raise a LookupError are in DATA_UNSAFE and are added back for arithmetic.
_STR = """capitalize casefold center count encode endswith expandtabs find index isalnum isalpha
    isascii isdecimal isdigit isidentifier islower isnumeric isprintable isspace istitle isupper
    join ljust lower lstrip partition removeprefix removesuffix replace rfind rindex rjust
    rpartition rsplit rstrip split splitlines startswith strip swapcase title upper zfill
    decode hex""".split()
_SEQ = ("append clear copy count extend index insert reverse sort "
        "appendleft extendleft rotate maxlen").split()
_MAP = ("fromkeys get items keys setdefault update values most_common elements "
        "subtract total").split()
_SET = """add difference difference_update discard intersection intersection_update isdisjoint
    issubset issuperset symmetric_difference symmetric_difference_update union""".split()
_NUM = ("bit_length bit_count conjugate real imag is_integer numerator denominator start "
        "stop step args name value").split()
_ARR = ("typecode itemsize tobytes tolist fromlist frombytes byteswap buffer_info fromunicode "
        "tounicode").split()
# Data attributes and decorators of property, partial, dataclasses.Field and the copy module.
_MISC = ("func keywords default default_factory fdel fget fset deleter getter "
         "setter mapping obj metadata kw_only init compare hash repr "
         "type error keyword register").split()
_SV = ("nondet_int nondet_bool nondet_str nondet_float nondet_float_range nondet_list "
       "nondet_dict check_type").split()

SV_VERIFIER_EXPORTS = frozenset(_SV) | {
    "random", "string", "collections", "types", "Any", "Callable", "Literal", "Union", "get_args",
    "get_origin"
}

_MODULE_NAMES = {
    "math":
    """pi e inf nan tau sqrt sin cos tan asin acos atan atan2 degrees radians hypot fabs gcd lcm
        copysign isnan isinf isfinite isclose fsum prod log2 log10 log1p dist modf frexp isqrt
        tanh asinh acosh atanh floor ceil trunc exp expm1 sinh cosh factorial comb perm ldexp
        erf erfc gamma lgamma""".split(),
    "random":
    "random randint randrange uniform shuffle seed sample getrandbits gauss normalvariate Random".
    split(),
    "itertools":
    """chain count cycle repeat product permutations combinations combinations_with_replacement
        accumulate groupby compress islice zip_longest takewhile dropwhile starmap tee filterfalse
        pairwise batched""".split(),
    "heapq":
    "heappush heapify nlargest nsmallest merge heappushpop".split(),
    "functools":
    "reduce partial lru_cache cache cmp_to_key total_ordering cached_property".split(),
    "collections":
    "Counter defaultdict deque OrderedDict".split(),
    "collections.abc":
    """Mapping MutableMapping Sequence MutableSequence Set MutableSet Iterable Iterator
        Callable Hashable Sized Container Collection Generator Reversible""".split(),
    "abc":
    "ABC ABCMeta abstractmethod".split(),
    "typing":
    """Any List Dict Set FrozenSet Tuple Type Optional Union Callable Iterable Iterator Sequence
        Mapping MutableMapping MutableSequence MutableSet Generator Literal Annotated ClassVar Final
        Generic Protocol Deque DefaultDict Counter OrderedDict Collection Container AbstractSet
        TypeVar NamedTuple TypedDict cast overload final TYPE_CHECKING NoReturn Never Self
        TypeAlias runtime_checkable TypeGuard""".split(),
    "dataclasses":
    "dataclass field".split(),
    "copy":
    "copy deepcopy".split(),
    "string":
    """ascii_letters ascii_lowercase ascii_uppercase digits hexdigits octdigits punctuation
        printable whitespace capwords""".split(),
    "array":
    "array".split(),
    "__future__":
    "annotations division print_function absolute_import unicode_literals generators".split(),
    "_sv_verifier":
    _SV,
}

# arithmetic: random.choice and friends cannot raise ZeroDivisionError.
# data-lookup: choice/choices raise IndexError on an empty sequence, and the
# heap pops raise IndexError.
_ARITH_ONLY_MODULE_NAMES = {
    "random": ["choice", "choices"],
    "heapq": ["heappop", "heapreplace"],
}

# Names that raise ZeroDivisionError for some argument.  Only a name that is
# also a method of an allowed type matters; they are simply never listed above.
# Names that raise KeyError or IndexError, added back for arithmetic.
# Library names that raise a LookupError (random.choice, heapq.heappop, ...).
LOOKUP_ATTRS = frozenset({"choice", "choices", "heappop", "heapreplace"})
DATA_UNSAFE_ATTRS = frozenset("pop popitem popleft remove move_to_end format format_map".split())

# OverflowError sites.  OverflowError is outside both families, so these are
# rejected on instruction (the family check does not need them) and reported
# separately so that the cost of the extra caution can be read off.
OVERFLOW_ATTRS = frozenset("floor ceil trunc exp expm1 sinh cosh factorial comb perm ldexp".split())
OVERFLOW_CALLS = frozenset("int float round complex".split())

# Names that raise KeyError or IndexError for some argument, found by the sampled audit
# (tools/audit_stdlib.py) or, for insert and shuffle, by reading after the audit missed them.
# Counter: Counter.__delitem__ hashes a key twice, which a stateful __hash__ turns into a KeyError.
DATA_UNSAFE_LATE = frozenset({
    "insert", "shuffle", "NamedTuple", "TypedDict", "sample", "heappushpop", "OrderedDict",
    "Counter"
})
# The collections.abc mixins hand the program's own object to C code through unbound calls.
ABC_MIXINS = frozenset("Mapping MutableMapping Sequence MutableSequence Set MutableSet "
                       "AbstractSet".split())
# Library classes written in Python: their methods run on whatever `self` the program passes.
LIBRARY_PY_CLASSES = frozenset(
    "Counter Random Enum IntEnum Flag IntFlag StrEnum ABC ABCMeta Generic Protocol".split())
# Library classes a class may inherit from, by the module that defines them, besides `object`,
# the builtin exceptions and classes of the program.  Enum and its relatives are not among them:
# EnumType's code looks members up in the program's own namespace and raises a KeyError.
LIB_BASE_MODULES = {"ABC": ("abc", ), "Generic": ("typing", ), "Protocol": ("typing", )}
GENERIC_BASES = frozenset({"Generic", "Protocol"})


def _build(family):
    mods = {m: set(ns) for m, ns in _MODULE_NAMES.items()}
    if family == "arithmetic":
        for m, ns in _ARITH_ONLY_MODULE_NAMES.items():
            mods[m] |= set(ns)
    else:
        for m in mods:
            mods[m] -= DATA_UNSAFE_LATE
    attrs = set(_STR + _SEQ + _MAP + _SET + _NUM + _ARR + _MISC)
    for ns in mods.values():
        attrs |= ns
    if family == "arithmetic":
        attrs |= DATA_UNSAFE_ATTRS
    else:
        attrs -= DATA_UNSAFE_ATTRS | DATA_UNSAFE_LATE
        attrs -= LOOKUP_ATTRS
    return mods, frozenset(attrs)


MODULE_NAMES = {}
SAFE_ATTRS = {}
for _f in FAMILIES:
    MODULE_NAMES[_f], SAFE_ATTRS[_f] = _build(_f)

ALLOWED_NODES = frozenset(
    """Module Expr Assign AugAssign AnnAssign Return Delete For While If With Try Assert Import
    ImportFrom Global Nonlocal Pass Break Continue FunctionDef ClassDef Lambda BoolOp
    NamedExpr BinOp UnaryOp IfExp Dict Set ListComp SetComp DictComp GeneratorExp Compare Call
    FormattedValue JoinedStr Constant Attribute Subscript Starred Name List Tuple Slice Yield
    YieldFrom comprehension ExceptHandler arguments arg keyword alias withitem Load Store Del
    And Or Add Sub Mult MatMult Div Mod Pow LShift RShift BitOr BitXor BitAnd FloorDiv Invert
    Not UAdd USub Eq NotEq Lt LtE Gt GtE Is IsNot In NotIn""".split())

if __package__:
    _UNIVERSE = importlib.import_module(f"{__package__}.exception_census_universe")
else:
    _UNIVERSE = importlib.import_module("exception_census_universe")
STDLIB_314, UNIVERSE_314 = _UNIVERSE.STDLIB_314, _UNIVERSE.UNIVERSE_314

UNIVERSE = UNIVERSE_314
STDLIB_NAMES = STDLIB_NAMES | STDLIB_314
# Imported by the interpreter itself at start-up when found on the path.
HIJACK_NAMES = frozenset({"sitecustomize", "usercustomize"})
