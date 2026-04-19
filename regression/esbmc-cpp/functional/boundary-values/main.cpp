#include <cassert>
#include <functional>
#include <climits>
#include <cfloat>

int main() {
    // Integer boundary tests
    std::hash<int> int_hasher;
    std::hash<long> long_hasher;
    std::hash<short> short_hasher;
    std::hash<char> char_hasher;
    
    // Test integer boundaries
    std::size_t hash_int_max = int_hasher(INT_MAX);
    std::size_t hash_int_min = int_hasher(INT_MIN);
    std::size_t hash_zero = int_hasher(0);
    std::size_t hash_neg_one = int_hasher(-1);
    std::size_t hash_one = int_hasher(1);
    
    // Test long boundaries
    std::size_t hash_long_max = long_hasher(LONG_MAX);
    std::size_t hash_long_min = long_hasher(LONG_MIN);
    
    // Test short boundaries
    std::size_t hash_short_max = short_hasher(SHRT_MAX);
    std::size_t hash_short_min = short_hasher(SHRT_MIN);
    
    // Test char boundaries
    std::size_t hash_char_max = char_hasher(CHAR_MAX);
    std::size_t hash_char_min = char_hasher(CHAR_MIN);
    
    // Determinism tests
    assert(int_hasher(INT_MAX) == hash_int_max);
    assert(int_hasher(INT_MIN) == hash_int_min);
    assert(long_hasher(LONG_MAX) == hash_long_max);
    assert(short_hasher(SHRT_MAX) == hash_short_max);
    assert(char_hasher(CHAR_MAX) == hash_char_max);
    
    // All should be valid
    assert(hash_int_max >= 0);
    assert(hash_int_min >= 0);
    assert(hash_long_max >= 0);
    assert(hash_short_max >= 0);
    assert(hash_char_max >= 0);
    
    return 0;
}
