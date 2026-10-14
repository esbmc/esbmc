# Regression for GitHub #5157 (HumanEval/103).
#
# Two frontend defects combined to fold a valid string comparison to False:
#   1. type_handler::get_typet handled hex()/oct() as char arrays but omitted
#      bin(), so `x = bin(n)` gave the symbol type `void` instead of `str`.
#      The void-typed symbol bypassed the string-comparison path.
#   2. A function with conflicting return types (`return -1` int sentinel plus
#      `return bin(...)` str) was narrowed to the first branch's type (int).
#      The call-site cross-type `==` fold then collapsed `f(...) == "..."` to
#      a constant False.


# Defect 1: assign bin()/hex()/oct() to a variable, then compare.
b = bin(3)
assert b == "0b11"
assert bin(-10) == "-0b1010"


# Defect 2: mixed int/str return type (the canonical HumanEval shape).
def rounded_avg(n, m):
    if m < n:
        return -1
    summation = 0
    for i in range(n, m + 1):
        summation += i
    return bin(round(summation / (m - n + 1)))


assert rounded_avg(1, 5) == "0b11"
assert rounded_avg(10, 20) == "0b1111"
