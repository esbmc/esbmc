#include <unordered_map>
#include <string>
#include <cassert>

int main() {
    std::unordered_map<int, std::string> m;
    
    // Test operator[] for insertion
    m[1] = "one";
    m[2] = "two";
    assert(m.size() == 2);
    assert(m[1] == "one");
    assert(m[2] == "two");

    // Test operator[] for modification
    m[1] = "ONE";
    assert(m[1] == "ONE");
    assert(m.size() == 2);

    // Test operator[] with default construction
    std::string& val = m[99];
    assert(val == ""); // Default constructed string
    assert(m.size() == 3);

    // Test at() for existing elements
    assert(m.at(1) == "ONE");
    assert(m.at(2) == "two");

    // Test insert_or_assign
    auto result1 = m.insert_or_assign(1, "one again");
    assert(result1.second == false); // Key existed
    assert(m[1] == "one again");

    auto result2 = m.insert_or_assign(50, "fifty");
    assert(result2.second == true); // New key
    assert(m[50] == "fifty");
    assert(m.size() == 4);

    return 0;
}

