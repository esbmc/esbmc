# len(C()) has to reach C.__len__ just as len(c) does. The dunder dispatch
# resolved the class from Name nodes only, so a constructor temporary found no
# class and len fell through to the builtin path, which measures the struct
# instead of calling __len__ -- a wrong length rather than an error.
#
# The dunder here reads no instance state: a method called on a constructor
# temporary does not see what __init__ wrote, which is a separate and older
# gap -- see method_on_temporary_loses_state.


class Sized:
    def __len__(self):
        return 3


class Shown:
    def __str__(self):
        return "shown"


assert len(Sized()) == 3
assert str(Shown()) == "shown"

# A named receiver kept working throughout, state and all.
s = Sized()
assert len(s) == 3

# The builtin containers still take the builtin path.
assert len([1, 2, 3]) == 3
assert len("abc") == 3
d = {"a": 1, "b": 2}
assert len(d) == 2
assert len((1, 2)) == 2
