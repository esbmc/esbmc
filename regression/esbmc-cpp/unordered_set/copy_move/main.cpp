#include <unordered_set>
#include <cassert>

int main() {
    std::unordered_set<int> s1;
    s1.insert(1);
    std::unordered_set<int> s2(s1);
    assert(s2.size() == 1);
    assert(s2.contains(1));
    assert(s1 == s2);
    return 0;
}

