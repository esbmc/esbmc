#include <unordered_set>
#include <cassert>

int main() {
    std::unordered_set<int> s;
    s.insert(10);
    s.insert(20);
    s.insert(30);
    
    // Test find existing elements
    auto it1 = s.find(20);
    assert(it1 != s.end());
    assert(*it1 == 20);
    
    // Test find non-existing element
    auto it2 = s.find(99);
    assert(it2 == s.end());
    
    // Test count
    assert(s.count(10) == 1);
    assert(s.count(99) == 0);
    
    // Test contains (C++20 feature)
    assert(s.contains(31) == true); // should fail
    assert(s.contains(99) == false);
    
    return 0;
}

