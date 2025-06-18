#include <vector>

// Dummy allocator
template<class T> struct my_alloc {};

int main() {
    // This line will fail with ESBMC's current vector implementation:
    // "error: too many template arguments for class template 'vector'"
    std::vector<int, my_alloc<int>> vec;
    
    return 0;
}
