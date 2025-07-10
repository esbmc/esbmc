#include <vector>

// Dummy allocator
template<class T> struct my_alloc {};

int main() {
    std::vector<int, my_alloc<int>> vec;
    
    return 0;
}
