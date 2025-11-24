#include <unordered_map>
#include <string>
#include <cassert>

int main() {
    std::unordered_map<int, std::string> m;

    // Test emplace
    auto result1 = m.emplace(42, "he");
    assert(result1.second == true);
    assert(m.size() == 1);
    assert(m.contains(42));
    assert(m[42] == "he");

    // Test emplace duplicate
    auto result2 = m.emplace(42, "wo");
    assert(result2.second == false);
    assert(m.size() == 1);
    assert(m[42] == "he"); // Original value preserved

    // Test try_emplace
    auto result3 = m.try_emplace(100, "hu");
    assert(result3.second == true);
    assert(m.size() == 2);
    assert(m.contains(100));
    assert(m[100] == "hu");

    // Test try_emplace duplicate
    auto result4 = m.try_emplace(100, "di");
    assert(result4.second == false);
    assert(m[100] == "hu"); // Original value preserved

    return 0;
}

