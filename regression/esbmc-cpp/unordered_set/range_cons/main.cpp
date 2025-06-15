#include <unordered_set>
#include <vector>
#include <cassert>

int main() {
    // Test range constructor
    std::vector<int> vec = {1, 2, 3, 2, 1}; // Duplicates should be ignored
    std::unordered_set<int> s1(vec.begin(), vec.end());
    assert(s1.size() == 3);
    assert(s1.contains(1));
    assert(s1.contains(2));
    assert(s1.contains(3));
    
    // Test range insertion
    std::unordered_set<int> s2;
    std::vector<int> vec2 = {4, 5, 6};
    s2.insert(vec2.begin(), vec2.end());
    assert(s2.size() == 3);
    assert(s2.contains(4));
    assert(s2.contains(5));
    assert(s2.contains(6));
    
    return 0;
}

