#include <unordered_map>
#include <string>
#include <cassert>

int main() {
    std::unordered_map<int, std::string> m;
    m.insert({10, "ten"});
    m.insert({20, "twenty"});
    m.insert({30, "thirty"});

    // Test find existing elements
    auto it1 = m.find(20);
    assert(it1 != m.end());
    assert(it1->first == 20);
    assert(it1->second == "twenty");

    // Test find non-existing element
    auto it2 = m.find(99);
    assert(it2 == m.end());

    // Test count
    assert(m.count(10) == 0); // should fail
    assert(m.count(99) == 0);

    // Test contains (C++20 feature)
    assert(m.contains(30) == true);
    assert(m.contains(99) == false);

    // Test operator[]
    assert(m[10] == "ten");
    assert(m[20] == "twenty");

    return 0;
}
