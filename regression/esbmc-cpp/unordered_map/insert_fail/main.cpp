#include <unordered_map>
#include <string>
#include <cassert>

int main() {
    std::unordered_map<int, std::string> m;

    // Test initial state
    assert(m.empty());
    assert(m.size() == 0);

    // Test insertion
    auto result = m.insert({42, "hello"});
    assert(result.second == true);  // Should be inserted
    assert(m.size() == 1);
    assert(!m.empty());

    // Test duplicate insertion
    auto result2 = m.insert({42, "world"});
    assert(result2.second == false); // Should not be inserted
    assert(m.size() == 2); // should fail!

    return 0;
}

