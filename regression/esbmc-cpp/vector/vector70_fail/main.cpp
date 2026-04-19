// Minimal test to verify compilation of vector with custom allocators
#include <vector>
#include <cstdlib>      // For malloc and free
#include <cstddef>      // For std::size_t and std::ptrdiff_t
#include <assert.h>

// Test 1: Absolutely minimal allocator
template <typename T>
struct min_alloc {
    using value_type = T;

    min_alloc() = default;
    template <class U> constexpr min_alloc(const min_alloc<U>&) noexcept {}

    T* allocate(std::size_t n) {
        return static_cast<T*>(malloc(n * sizeof(T)));
    }

    void deallocate(T* p, std::size_t) noexcept {
        free(p);
    }

    template <class U>
    struct rebind { using other = min_alloc<U>; };
};

// Test 2: Allocator with all standard typedefs
template <typename T>
struct full_alloc {
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    full_alloc() = default;
    template <class U> constexpr full_alloc(const full_alloc<U>&) noexcept {}

    template <class U>
    struct rebind { using other = full_alloc<U>; };

    T* allocate(std::size_t n) {
        return static_cast<T*>(malloc(n * sizeof(T)));
    }

    void deallocate(T* p, std::size_t) noexcept {
        free(p);
    }

    bool operator==(const full_alloc&) const noexcept { return true; }
    bool operator!=(const full_alloc&) const noexcept { return false; }
};

int main() {
    // These should all compile without "too many template arguments" error
    std::vector<int> vec1;                                 // Default allocator
    std::vector<int, std::allocator<int>> vec2;            // Explicit std::allocator
    std::vector<int, min_alloc<int>> vec3;                 // Minimal custom allocator
    std::vector<int, full_alloc<int>> vec4;                // Full custom allocator

    // Test basic functionality
    vec3.push_back(42);
    vec4.push_back(123);
    int res = (vec3[0] == 42 && vec4[0] == 123) ? 0 : 1;
    assert(res);
    return res;
}

