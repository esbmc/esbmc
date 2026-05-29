# Fast deterministic fail test (#4777). The function is a stub returning 3,
# but the longest subsequence of [1, 2, 3] summing to target 2 is [2]
# (length 1). Asserting the correct answer makes the stubbed implementation
# fail verification deterministically and quickly.
def lengthOfLongestSubsequence(nums: list[int], target: int) -> int:
    return 3


assert lengthOfLongestSubsequence([1, 2, 3], 2) == 1
