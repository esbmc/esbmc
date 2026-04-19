#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <cassert>

int minimumCoins_v1(const std::vector<int>& prices) {
    int n = prices.size();
    if (n == 0) return 0;
    if (n == 1) return prices[0];
    
    std::vector<int> dp(n, INT_MAX);
    
    // Initialize: buying item 0 gives us items 0 and 1
    for (int j = 0; j < 2; j++) {
        dp[j] = prices[0];
    }
    
    for (int i = 1; i < n; i++) {
        int price = dp[i - 1] + prices[i];
        // When we buy item i, we get items from i to min(n-1, i+i)
        for (int j = i; j < std::min(n, (i + 1) * 2); j++) {
            dp[j] = std::min(dp[j], price);
        }
    }
    
    return dp[n - 1];
}

int main() {
    // Test the original case
    std::vector<int> test1 = {1, 2, 3};
    int result1 = minimumCoins_v1(test1);
    std::cout << "minimumCoins_v1([1, 2, 3]) = " << result1 << std::endl;
    assert(result1 == 3);
    std::cout << "Test 1 passed!" << std::endl;
    
    // Additional test cases
    std::vector<int> test2 = {3, 1, 2};
    std::cout << "minimumCoins_v1([3, 1, 2]) = " << minimumCoins_v1(test2) << std::endl;
    
    std::vector<int> test3 = {1, 10, 1, 1};
    std::cout << "minimumCoins_v1([1, 10, 1, 1]) = " << minimumCoins_v1(test3) << std::endl;
    
    std::vector<int> test4 = {1};
    std::cout << "minimumCoins_v1([1]) = " << minimumCoins_v1(test4) << std::endl;
    
    std::vector<int> test5 = {};
    std::cout << "minimumCoins_v1([]) = " << minimumCoins_v1(test5) << std::endl;
    
    // Demonstrate what buying each item gives us
    std::cout << "\nWhat buying each item gives us for array of length 4:" << std::endl;
    for (int i = 0; i < 4; i++) {
        int start = i;
        int end = std::min(4, (i + 1) * 2);
        std::cout << "Buying item " << i << " gives items: [";
        for (int j = start; j < end; j++) {
            std::cout << j;
            if (j < end - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    return 0;
}
