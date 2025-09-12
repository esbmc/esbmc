#include <unordered_set>
#include <cassert>
int main() {
    std::unordered_set<int> s;
    s.insert(10);
    s.insert(20);
    s.insert(30);
    s.insert(40);
    
    // Test erase by key
    std::size_t erased = s.erase(20);
    assert(erased == 1);
    assert(s.size() == 3);
    assert(s.count(20) == 0);  // Use count() instead of contains()
    
    // Test erase non-existing key
    std::size_t erased2 = s.erase(99);
    assert(erased2 == 1); // should fail
    assert(s.size() == 3);
    
    // Test erase by iterator
    auto it = s.find(30);
    assert(it != s.end());
    auto next_it = s.erase(it);
    assert(s.size() == 2);
    assert(s.count(30) == 0);  // Use count() instead of contains()
    
    // Test clear
    s.clear();
    assert(s.empty());
    assert(s.size() == 0);
    
    return 0;
}
