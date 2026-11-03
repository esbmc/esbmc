# A conditional expression binds a str on one branch and an int on the other,
# so `v + 1` is a TypeError on the str branch. The if/else spelling of the same
# code already reported it; this spelling verified instead (#8263).
def main():
    i = nondet_int()
    if not (0 <= i and i <= 1):
        return
    v = "s" if i == 0 else 1
    y = v + 1


main()
