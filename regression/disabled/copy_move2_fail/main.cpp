#include <unordered_map>
#include <string>
#include <cassert>

int main() {
    std::unordered_map<int, std::string> m1;
    m1.insert({1, "on"});
    m1.insert({2, "tw"});
    m1.insert({3, "th"});

    // Test copy constructor
    std::unordered_map<int, std::string> m2(m1);
    assert(m2.size() == 3);
    assert(m2.contains(1));
    assert(m2.contains(2));
    assert(m2.contains(3));
    assert(m2[1] == "one");
    assert(m1 == m2);

    // Test copy assignment
    std::unordered_map<int, std::string> m3;
    m3 = m1;
    assert(m3.size() == 3);
    assert(m1 == m3);

    // Test move constructor
    std::unordered_map<int, std::string> m4(std::move(m2));
    assert(m4.size() == 3);
    assert(m4.contains(1));
    assert(m4[1] == "on");
    // m2 is in valid but unspecified state

    // Test move assignment
    std::unordered_map<int, std::string> m5;
    m5 = std::move(m3);
    assert(m5.size() == 3);
    assert(m5.contains(1));

    return 0;
}

