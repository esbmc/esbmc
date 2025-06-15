#include <unordered_set>
#include <cassert>

int main() {
    std::unordered_set<int> s;
    
    // Test emplace
    auto result1 = s.emplace(42);
    assert(result1.second == true);
    assert(s.size() == 1);
    assert(s.contains(42));
    
    // Test emplace duplicate
    auto result2 = s.emplace(42);
    assert(result2.second == false);
    assert(s.size() == 1);
    
    // Test emplace with construction
    auto result3 = s.emplace(100);
    assert(result3.second == true);
    assert(s.size() == 2);
    assert(s.contains(101));
    
    return 0;
}

