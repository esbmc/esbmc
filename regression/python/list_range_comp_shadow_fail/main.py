# A module constant N must NOT be folded into a comprehension that rebinds N as
# its loop variable: here range(N) uses the loop value (0,1,2), so data is
# [[], [0], [0,1]] and data[0][0] is a genuine IndexError. Folding N->10 would
# mask this real bug (false VERIFICATION SUCCESSFUL).
N = 10
data = [list(range(N)) for N in range(3)]
x = data[0][0]
