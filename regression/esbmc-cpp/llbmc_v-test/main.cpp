#include <cassert>
#include <vector>

int main()
{
    std::vector<int> v;

    for (int i = 0; i < 10; ++i) {
        v.push_back(i);
    }

    for (int i = 0 ; i < 10; ++i) {
        assert(v[i] == i);
    }

    return 0;
}
