#include <cassert>
#include <functional>

int main() {
    std::hash<int> hasher;
    
    // Test powers of 2 (common edge cases for hash functions)
    std::size_t hash_1 = hasher(1);          // 2^0
    std::size_t hash_2 = hasher(2);          // 2^1
    std::size_t hash_4 = hasher(4);          // 2^2
    std::size_t hash_8 = hasher(8);          // 2^3
    std::size_t hash_16 = hasher(16);        // 2^4
    std::size_t hash_32 = hasher(32);        // 2^5
    std::size_t hash_64 = hasher(64);        // 2^6
    std::size_t hash_128 = hasher(128);      // 2^7
    std::size_t hash_256 = hasher(256);      // 2^8
    std::size_t hash_512 = hasher(512);      // 2^9
    std::size_t hash_1024 = hasher(1024);    // 2^10
    std::size_t hash_2048 = hasher(2048);    // 2^11
    std::size_t hash_4096 = hasher(4096);    // 2^12
    
    // Test determinism for powers of 2
    assert(hasher(1) == hash_1);
    assert(hasher(2) == hash_2);
    assert(hasher(16) == hash_16);
    assert(hasher(256) == hash_256);
    assert(hasher(1024) == hash_1024);
    assert(hasher(4096) == hash_4096);
    
    // Test negative powers of 2
    std::size_t hash_neg_1 = hasher(-1);
    std::size_t hash_neg_2 = hasher(-2);
    std::size_t hash_neg_4 = hasher(-4);
    std::size_t hash_neg_8 = hasher(-8);
    std::size_t hash_neg_16 = hasher(-16);
    std::size_t hash_neg_32 = hasher(-32);
    std::size_t hash_neg_64 = hasher(-64);
    std::size_t hash_neg_128 = hasher(-128);
    std::size_t hash_neg_256 = hasher(-256);
    std::size_t hash_neg_512 = hasher(-512);
    std::size_t hash_neg_1024 = hasher(-1024);
    
    // Test determinism for negative powers of 2
    assert(hasher(-1) == hash_neg_1);
    assert(hasher(-2) == hash_neg_2);
    assert(hasher(-16) == hash_neg_16);
    assert(hasher(-256) == hash_neg_256);
    assert(hasher(-1024) == hash_neg_1024);
    
    // Test powers of 2 minus 1 (also common edge cases)
    std::size_t hash_3 = hasher(3);          // 2^2 - 1
    std::size_t hash_7 = hasher(7);          // 2^3 - 1
    std::size_t hash_15 = hasher(15);        // 2^4 - 1
    std::size_t hash_31 = hasher(31);        // 2^5 - 1
    std::size_t hash_63 = hasher(63);        // 2^6 - 1
    std::size_t hash_127 = hasher(127);      // 2^7 - 1
    std::size_t hash_255 = hasher(255);      // 2^8 - 1
    std::size_t hash_511 = hasher(511);      // 2^9 - 1
    std::size_t hash_1023 = hasher(1023);    // 2^10 - 1
    
    // Test determinism for powers of 2 minus 1
    assert(hasher(3) == hash_3);
    assert(hasher(7) == hash_7);
    assert(hasher(15) == hash_15);
    assert(hasher(31) == hash_31);
    assert(hasher(127) == hash_127);
    assert(hasher(255) == hash_255);
    assert(hasher(1023) == hash_1023);
    
    // Test unsigned int with large powers of 2
    std::hash<unsigned int> uint_hasher;
    
    std::size_t hash_65536 = uint_hasher(65536U);      // 2^16
    std::size_t hash_131072 = uint_hasher(131072U);    // 2^17
    std::size_t hash_262144 = uint_hasher(262144U);    // 2^18
    std::size_t hash_524288 = uint_hasher(524288U);    // 2^19
    std::size_t hash_1048576 = uint_hasher(1048576U);  // 2^20
    
    // Test determinism
    assert(uint_hasher(65536U) == hash_65536);
    assert(uint_hasher(1048576U) == hash_1048576);
    
    // Test zero (which is also a power of 2 in some sense: 2^0 * 0)
    std::size_t hash_zero = hasher(0);
    assert(hasher(0) == hash_zero);
    
    // All hashes should be valid
    assert(hash_1 >= 0);
    assert(hash_256 >= 0);
    assert(hash_neg_128 >= 0);
    assert(hash_65536 >= 0);
    assert(hash_zero >= 0);
    
    return 0;
}
