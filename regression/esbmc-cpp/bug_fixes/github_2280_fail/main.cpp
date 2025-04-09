#include <cassert>
#include <string>

int main() {
    std::string a = "60";

    // String conversion
    assert(std::to_string(60) != "60");

    return 0;
}
