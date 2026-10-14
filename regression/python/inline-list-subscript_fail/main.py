# Negative variant: the inline subscript must still be able to fail (no
# spurious SUCCESSFUL). sorted([3,1,2])[0] is 1, not 99.


def smallest(nums):
    return sorted(nums)[0]


assert smallest([3, 1, 2]) == 99
