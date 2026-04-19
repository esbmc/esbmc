#include <unordered_set>
#include <cassert>

int main() {
    std::unordered_set<int> s;
    
    // Test max_size
    assert(s.max_size() > 0);
    assert(s.max_size() >= s.size());
    
    // Test with zero bucket count constructor
    std::unordered_set<int> s2(0); // Should default to minimum
    assert(s2.bucket_count() >= 0);
    
    // Test hash function and key_eq access
    auto hasher = s.hash_function();
    auto key_eq = s.key_eq();
    
    // Basic functionality should work
    std::size_t hash1 = hasher(42);
    std::size_t hash2 = hasher(42);
    assert(hash1 == hash2);
    
    assert(key_eq(42, 42) == true);
    assert(key_eq(42, 43) == false);
    
    return 0;
}

