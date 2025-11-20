#include <cassert>
#include <functional>
#include <set>

int main() {
    std::hash<int> hasher;
    std::set<std::size_t> hash_values;
    
    // Test consecutive integers (should have good distribution)
    for(int i = 0; i < 100; ++i) {
        std::size_t hash_val = hasher(i);
        assert(hash_val >= 0);
        
        // Check determinism
        assert(hasher(i) == hash_val);
        
        // Collect for distribution check
        hash_values.insert(hash_val);
    }
    
    // Test that we don't have too many collisions
    // (This is probabilistic, but with 100 values we should have
    // a reasonable number of unique hashes)
    assert(hash_values.size() > 50);  // At least 50% unique
    
    // Test some specific patterns that might cause collisions
    int pattern1 = 0x12345678;
    int pattern2 = 0x87654321;
    int pattern3 = 0x11111111;
    int pattern4 = 0x22222222;
    
    std::size_t hash_p1 = hasher(pattern1);
    std::size_t hash_p2 = hasher(pattern2);
    std::size_t hash_p3 = hasher(pattern3);
    std::size_t hash_p4 = hasher(pattern4);
    
    // These patterns should ideally produce different hashes
    assert(hash_p1 != hash_p2);
    assert(hash_p3 != hash_p4);
    
    // Test bit patterns that might be problematic
    assert(hasher(0x00000000) != hasher(0xFFFFFFFF));
    assert(hasher(0x55555555) != hasher(0xAAAAAAAA));
    
    return 0;
}

