#include <cassert>
#include <functional>
#include <climits>

int main() {
    std::hash<long long> ll_hasher;
    std::hash<unsigned long long> ull_hasher;
    std::hash<size_t> size_hasher;
    
    // Test very large numbers
    long long large_positive = LLONG_MAX;
    long long large_negative = LLONG_MIN;
    unsigned long long max_unsigned = ULLONG_MAX;
    
    std::size_t hash_large_pos = ll_hasher(large_positive);
    std::size_t hash_large_neg = ll_hasher(large_negative);
    std::size_t hash_max_unsigned = ull_hasher(max_unsigned);
    
    // Test numbers near overflow boundaries
    std::size_t hash_near_max1 = ll_hasher(LLONG_MAX - 1);
    std::size_t hash_near_max2 = ll_hasher(LLONG_MAX - 2);
    std::size_t hash_near_min1 = ll_hasher(LLONG_MIN + 1);
    std::size_t hash_near_min2 = ll_hasher(LLONG_MIN + 2);
    
    // Test size_t boundaries
    std::size_t hash_size_max = size_hasher(SIZE_MAX);
    std::size_t hash_size_zero = size_hasher(0);
    
    // Test determinism
    assert(ll_hasher(LLONG_MAX) == hash_large_pos);
    assert(ll_hasher(LLONG_MIN) == hash_large_neg);
    assert(ull_hasher(ULLONG_MAX) == hash_max_unsigned);
    assert(size_hasher(SIZE_MAX) == hash_size_max);
    
    // Test that boundary values produce different hashes
    assert(hash_large_pos != hash_large_neg);
    assert(hash_near_max1 != hash_near_max2);
    assert(hash_near_min1 != hash_near_min2);
    
    // All should be valid
    assert(hash_large_pos >= 0);
    assert(hash_large_neg >= 0);
    assert(hash_max_unsigned >= 0);
    assert(hash_size_max >= 0);
    
    return 0;
}

