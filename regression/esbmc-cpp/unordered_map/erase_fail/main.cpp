#include <unordered_map>
#include <string>
#include <cassert>

int main() {
    std::unordered_map<int, int> m;
    m.insert({10, 1});
    m.insert({20, 2});
    m.insert({30, 3});
    m.insert({40, 4});

    // Test erase by key
    std::size_t erased = m.erase(20);
    assert(erased == 1);
    assert(m.size() == 3);
    assert(m.count(20) == 0);

    // Test erase non-existing key
    std::size_t erased2 = m.erase(99);
    assert(erased2 == 0);
    assert(m.size() == 3);
    
    // Test erase by iterator
    auto it = m.find(30);
    assert(it != m.end());
    auto next_it = m.erase(it);
    assert(m.size() == 2);
    assert(m.count(30) == 0);

    // Test clear
    m.clear();
    assert(!m.empty());
    assert(m.size() == 0);
    
    return 0;
}

