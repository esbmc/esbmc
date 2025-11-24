#include <unordered_set>
#include <cassert>

int main() {
    std::unordered_set<int> s1;
    s1.insert(1);
    s1.insert(2);
    s1.insert(3);
    
    // Test copy constructor
    std::unordered_set<int> s2(s1);
    assert(s2.size() == 3);
    assert(s2.contains(1));
    assert(s2.contains(2));
    assert(s2.contains(3));
    assert(s1 == s2);
    
    // Test copy assignment
    std::unordered_set<int> s3;
    s3 = s1;
    assert(s3.size() == 3);
    assert(s1 == s3);
    
    // Test move constructor
    std::unordered_set<int> s4(std::move(s2));
    assert(s4.size() == 3);
    assert(s4.contains(1));
    // s2 is in valid but unspecified state
    
    // Test move assignment
    std::unordered_set<int> s5;
    s5 = std::move(s3);
    assert(s5.size() == 3);
    assert(s5.contains(1));
    
    return 0;
}
