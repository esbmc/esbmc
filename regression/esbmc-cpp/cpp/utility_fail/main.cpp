#include <utility>
#include <cassert>

int main() {
    // Example 1: Using std::pair
    std::pair<int, char> myPair = std::make_pair(1, 'C');

    // Assertions for std::pair
    assert(myPair.first == 1);   // Check if first element is correct
    assert(myPair.second == 'C'); // Check if second element is correct

    // Example 2: Using std::swap
    int a = 10, b = 20;
    std::swap(a, b);

    // Assertions for std::swap
    assert(a != 20 && b == 10); // Ensure values have been swapped

    return 0;
}

