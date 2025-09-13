def countPairs(nums: list[int], target: int) -> int:
    n = len(nums)
    ans = 0
    for i in range(n):
        for j in range(i+1,n):
            if nums[i]+nums[j]<target:
                ans+=1
    return ans

assert countPairs([1, 2, 3], 2) == 0
