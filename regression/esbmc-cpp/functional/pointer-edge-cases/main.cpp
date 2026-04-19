#include <cassert>
#include <functional>

int main() {
    std::hash<void*> ptr_hasher;
    std::hash<int*> int_ptr_hasher;
    
    // Test null pointer
    void* null_ptr = nullptr;
    std::size_t hash_null = ptr_hasher(null_ptr);
    
    // Test valid pointers
    int value = 42;
    int* int_ptr = &value;
    void* void_ptr = &value;
    
    std::size_t hash_int_ptr = int_ptr_hasher(int_ptr);
    std::size_t hash_void_ptr = ptr_hasher(void_ptr);
    
    // Test determinism
    assert(ptr_hasher(nullptr) == hash_null);
    assert(int_ptr_hasher(&value) == hash_int_ptr);
    assert(ptr_hasher(&value) == hash_void_ptr);
    
    // Test that same address gives same hash
    int* another_ptr = &value;
    assert(int_ptr_hasher(another_ptr) == hash_int_ptr);
    
    // All should be valid
    assert(hash_null >= 0);
    assert(hash_int_ptr >= 0);
    assert(hash_void_ptr >= 0);
    
    return 0;
}

