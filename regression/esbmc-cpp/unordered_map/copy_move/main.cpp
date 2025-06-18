#include <unordered_map>
#include <string>
#include <cassert>

int main() {
    std::unordered_map<int, std::string> m1;
    m1.insert({1, "one"});
    std::unordered_map<int, std::string> m2(m1);
    assert(m2.size() == 1);
    assert(m2.contains(1));
    assert(m2[1] == "one");
    assert(m1 == m2);
    return 0;
}

