#include <vector>

// Dummy allocator
template<class T> struct my_alloc {};

int main() {
<<<<<<< HEAD
<<<<<<< HEAD
=======
    // This line will fail with ESBMC's current vector implementation:
    // "error: too many template arguments for class template 'vector'"
>>>>>>> 03f9c6079 ([regression] added test cases for vector)
=======
>>>>>>> 06b470233 (Update main.cpp)
    std::vector<int, my_alloc<int>> vec;
    
    return 0;
}
