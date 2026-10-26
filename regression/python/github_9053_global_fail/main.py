# GitHub #9053: a module-level store changes the element the literal was built
# with, though the function reading it only declares the list global.
dp = [0, 0, 0]
dp[1] = 1.5


def g(b):
    global dp
    return dp[b]


assert g(1) == 1
