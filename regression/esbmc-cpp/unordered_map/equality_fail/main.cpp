#include <unordered_map>
#include <cassert>

int main() {
    std::unordered_map<int, int> m1;
    std::unordered_map<int, int> m2;

    // Test empty maps equality
    assert(m1 == m2);
    assert(!(m1 != m2));

    // Test after insertion
    m1[1] = 1;
    m1[2] = 2;
    
    m2[2] = 2;
    m2[1] = 1;

    assert(m1 == m2); // Order shouldn't matter
    assert(!(m1 != m2));

    // Test different values
    m2[1] = 3;
    assert(!(m1 == m2));
    assert(m1 != m2);

    // Test different sizes
    m2[1] = 1; // Fix the value
    m2[3] = 3;
    assert(!(m1 == m2));
    assert(m1 == m2); // should fail

    return 0;
}
