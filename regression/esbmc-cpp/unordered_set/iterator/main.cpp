#include <unordered_set>
#include <cassert>

int main() {
    std::unordered_set<int> s;
    s.insert(1);
    s.insert(2);
    s.insert(3);
    
    // Test iterator traversal
    int count = 0;
    int sum = 0;
    for (auto it = s.begin(); it != s.end(); ++it) {
        count++;
        sum += *it;
    }
    
    assert(count == 3);
    assert(sum == 6); // 1 + 2 + 3
    
    // Test range-based for loop
    int count2 = 0;
    for (const auto& value : s) {
        count2++;
        assert(s.contains(value));
    }
    assert(count2 == 3);
    
    // Test const iterators
    const auto& cs = s;
    auto cit = cs.cbegin();
    assert(cit != cs.cend());
    
    return 0;
}

