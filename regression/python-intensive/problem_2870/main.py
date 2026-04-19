def alternatingSubarray(nums: list[int]) -> int:
    n = len(nums)
    ans = -1
    for i in range(n):
        for j in range(i + 1, n):
            if nums[j] != nums[i] + ((j - i) & 1):
                break
            ans = max(ans, j - i + 1)
    return ans

assert alternatingSubarray([1, 2, 3]) == 2
