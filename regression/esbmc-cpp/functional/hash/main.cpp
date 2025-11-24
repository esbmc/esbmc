#include <cassert>
#include <functional>

int main() {
    std::hash<int> hasher;
    
    // Test sequential values around zero
    for (int i = -10; i <= 10; i++) {
        std::size_t hash_i = hasher(i);
        std::size_t hash_i_repeat = hasher(i);
        
        // Test determinism for each value
        assert(hash_i == hash_i_repeat);
        assert(hash_i >= 0);  // All hashes should be valid
        
        // Test multiple calls for same value
        for (int j = 0; j < 3; j++) {
            assert(hasher(i) == hash_i);
        }
    }
    
    // Test larger sequential values
    for (int i = 100; i <= 110; i++) {
        std::size_t hash_i = hasher(i);
        assert(hasher(i) == hash_i);  // Determinism
        assert(hash_i >= 0);
    }
    
    // Test negative sequential values
    for (int i = -110; i <= -100; i++) {
        std::size_t hash_i = hasher(i);
        assert(hasher(i) == hash_i);  // Determinism
        assert(hash_i >= 0);
    }
    
    // Test repeated calls with same value (stress test)
    int stress_value = 12345;
    std::size_t stress_hash = hasher(stress_value);
    
    for (int i = 0; i < 20; i++) {
        assert(hasher(stress_value) == stress_hash);
    }
    
    // Test char sequential values
    std::hash<char> char_hasher;
    
    // Test ASCII digit range
    for (char c = '0'; c <= '9'; c++) {
        std::size_t hash_c = char_hasher(c);
        assert(char_hasher(c) == hash_c);
        assert(hash_c >= 0);
    }
    
    // Test ASCII uppercase range
    for (char c = 'A'; c <= 'Z'; c++) {
        std::size_t hash_c = char_hasher(c);
        assert(char_hasher(c) == hash_c);
        assert(hash_c >= 0);
    }
    
    // Test ASCII lowercase range
    for (char c = 'a'; c <= 'z'; c++) {
        std::size_t hash_c = char_hasher(c);
        assert(char_hasher(c) == hash_c);
        assert(hash_c >= 0);
    }
    
    // Test bool stress (limited values but multiple calls)
    std::hash<bool> bool_hasher;
    
    for (int i = 0; i < 10; i++) {
        assert(bool_hasher(true) == 1);
        assert(bool_hasher(false) == 0);
        assert(bool_hasher(true) != bool_hasher(false));
    }
    
    // Test unsigned int with sequential values
    std::hash<unsigned int> uint_hasher;
    
    for (unsigned int i = 0U; i <= 20U; i++) {
        std::size_t hash_i = uint_hasher(i);
        assert(uint_hasher(i) == hash_i);
        assert(hash_i >= 0);
    }
    
    // Test large unsigned values
    unsigned int large_base = 1000000U;
    for (unsigned int i = large_base; i <= large_base + 10U; i++) {
        std::size_t hash_i = uint_hasher(i);
        assert(uint_hasher(i) == hash_i);
        assert(hash_i >= 0);
    }
    
    // Test pointer stress with null
    std::hash<int*> ptr_hasher;
    
    for (int i = 0; i < 5; i++) {
        assert(ptr_hasher(nullptr) == ptr_hasher(nullptr));
    }
    
    // Test multiple pointers to same value
    int shared_val = 999;
    int* ptr1 = &shared_val;
    int* ptr2 = &shared_val;
    
    for (int i = 0; i < 5; i++) {
        assert(ptr_hasher(ptr1) == ptr_hasher(ptr2));
        assert(ptr_hasher(&shared_val) == ptr_hasher(ptr1));
    }
    
    return 0;
}
