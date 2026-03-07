z0 = complex()
assert z0.real == 0.0
assert z0.imag == 0.0

z1 = complex(1)
assert z1.real == 1.0
assert z1.imag == 0.0

z2 = complex(1, 2)
assert z2.real == 1.0
assert z2.imag == 2.0

z3 = complex(True, False)
assert z3.real == 1.0
assert z3.imag == 0.0

c1 = complex(1, 2)
z4 = complex(c1)
assert z4.real == 1.0
assert z4.imag == 2.0

z5 = complex(c1, 3)
assert z5.real == 1.0
assert z5.imag == 5.0

c2 = complex(2, 3)
z6 = complex(1, c2)
assert z6.real == -2.0
assert z6.imag == 2.0

z7 = complex(c1, c2)
assert z7.real == -2.0
assert z7.imag == 4.0

# String accepted in one-argument form
z8 = complex("1")
assert z8.real == 1.0
assert z8.imag == 0.0

z9 = complex("-3.5")
assert z9.real == -3.5
assert z9.imag == 0.0

z10 = complex("  1  ")
assert z10.real == 1.0
assert z10.imag == 0.0

z11 = complex("3+4j")
assert z11.real == 3.0
assert z11.imag == 4.0

z11b = complex("(3+4j)")
assert z11b.real == 3.0
assert z11b.imag == 4.0

z11d = complex(" -j ")
assert z11d.real == 0.0
assert z11d.imag == -1.0

z11f = complex("j")
assert z11f.real == 0.0
assert z11f.imag == 1.0

z11g = complex("+j")
assert z11g.real == 0.0
assert z11g.imag == 1.0

z11h = complex("1e2+3e-1j")
assert z11h.real == 100.0
assert z11h.imag == 0.3

z11i = complex(" (3+4j) ")
assert z11i.real == 3.0
assert z11i.imag == 4.0

# non-literal string symbol
s_complex = "5+6j"
z11e = complex(s_complex)
assert z11e.real == 5.0
assert z11e.imag == 6.0

# CPython-compatible: bytearray is rejected
raised_type = False
try:
    complex(b"7+8j")
except TypeError:
    raised_type = True
assert raised_type

# bytes variable form must also be rejected
b_value: bytes = b"7+8j"
raised_type = False
try:
    complex(b_value)
except TypeError:
    raised_type = True
assert raised_type

raised_type = False
try:
    complex(bytearray(b"-3.5"))
except TypeError:
    raised_type = True
assert raised_type

# kwargs
z12 = complex(real=1, imag=2)
assert z12.real == 1.0
assert z12.imag == 2.0

z12b = complex(real=1)
assert z12b.real == 1.0
assert z12b.imag == 0.0

z12c = complex(imag=2)
assert z12c.real == 0.0
assert z12c.imag == 2.0

z12e = complex(imag=2, real=1)
assert z12e.real == 1.0
assert z12e.imag == 2.0

z12d = complex(real=c1, imag=3)
assert z12d.real == 1.0
assert z12d.imag == 5.0

z12f = complex(real="1")
assert z12f.real == 1.0
assert z12f.imag == 0.0

# malformed string -> ValueError
raised_value = False
try:
    complex("x")
except ValueError:
    raised_value = True
assert raised_value

# internal spaces in complex string -> ValueError (CPython behavior)
raised_value = False
try:
    complex("1 + 2j")
except ValueError:
    raised_value = True
assert raised_value

raised_value = False
try:
    complex("++1j")
except ValueError:
    raised_value = True
assert raised_value

# second arg string -> TypeError
raised_type = False
try:
    complex(1, "2")
except TypeError:
    raised_type = True
assert raised_type

raised_type = False
try:
    complex(1, b"2")
except TypeError:
    raised_type = True
assert raised_type

# too many args -> TypeError
raised_type = False
try:
    complex(1, 2, 3)
except TypeError:
    raised_type = True
assert raised_type

# positional + duplicate keyword -> TypeError
raised_type = False
try:
    complex(1, real=2)
except TypeError:
    raised_type = True
assert raised_type

# duplicate imag via positional + keyword -> TypeError
raised_type = False
try:
    complex(1, 2, imag=3)
except TypeError:
    raised_type = True
assert raised_type

# unexpected keyword -> TypeError
raised_type = False
try:
    complex(foo=1)
except TypeError:
    raised_type = True
assert raised_type


class HasComplex:
    def __complex__(self) -> complex:
        return complex(1, 2)


class HasFloat:
    def __float__(self) -> float:
        return 3.5


class HasIndex:
    def __index__(self) -> int:
        return 4


z13 = complex(HasComplex())
assert z13.real == 1.0
assert z13.imag == 2.0

z14 = complex(HasFloat())
assert z14.real == 3.5
assert z14.imag == 0.0

z15 = complex(HasIndex())
assert z15.real == 4.0
assert z15.imag == 0.0


class BadComplex:
    def __complex__(self) -> float:
        return 1.0


raised_type = False
try:
    complex(BadComplex())
except TypeError:
    raised_type = True
assert raised_type


class BadFloat:
    def __float__(self) -> str:
        return "3.0"


raised_type = False
try:
    complex(BadFloat())
except TypeError:
    raised_type = True
assert raised_type


class BadIndex:
    def __index__(self) -> str:
        return "2"


raised_type = False
try:
    complex(BadIndex())
except TypeError:
    raised_type = True
assert raised_type


class HasComplexAndFloat:
    def __complex__(self) -> complex:
        return complex(2, 3)

    def __float__(self) -> float:
        return 9.0


z16 = complex(HasComplexAndFloat())
assert z16.real == 2.0
assert z16.imag == 3.0
