#include <cassert>
#include <string>

int main() {
    std::string a = "60";
    std::string b = "5";

    // Integer conversion
    assert(std::stoi(a) == 60);
    assert(std::stoi(b) == 5);

    // Float conversion
    assert(std::stof(a) == 60.0f);
    assert(std::stof(b) == 5.0f);

    return 0;
}
