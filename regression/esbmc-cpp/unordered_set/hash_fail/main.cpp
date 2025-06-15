#include <unordered_set>
#include <cassert>

int main() {
    std::unordered_set<int> s;
    
    // Test initial bucket properties
    assert(s.bucket_count() >= 8); // should fail
    assert(s.load_factor() == 0.0);
    
    // Insert elements and test bucket properties
    s.insert(1);
    s.insert(2);
    s.insert(3);
    
    assert(s.load_factor() > 0.0);
    assert(s.load_factor() <= s.max_load_factor());
    
    // Test bucket function
    std::size_t bucket1 = s.bucket(1);
    std::size_t bucket2 = s.bucket(1); // Should be same
    assert(bucket1 == bucket2);
    assert(bucket1 < s.bucket_count());
    
    // Test reserve
    std::size_t old_bucket_count = s.bucket_count();
    s.reserve(100);
    assert(s.bucket_count() >= old_bucket_count);
    assert(s.size() == 3); // Size unchanged
    
    return 0;
}

