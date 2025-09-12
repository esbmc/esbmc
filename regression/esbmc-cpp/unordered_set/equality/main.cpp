#include <unordered_set>
#include <cassert>

int main() {
    std::unordered_set<int> s;
    
    // Test initial state
    assert(s.empty());
    assert(s.size() == 0);
    
    // Test insertion
    auto result = s.insert(42);
    assert(result.second == true);  // Should be inserted
    assert(s.size() == 1);
    assert(!s.empty());
    
    // Test duplicate insertion
    auto result2 = s.insert(42);
    assert(result2.second == false); // Should not be inserted
    assert(s.size() == 1);
    
    return 0;
}

