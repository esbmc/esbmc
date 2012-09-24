#include <cassert>
#include <queue>

int main()
{
    std::queue<int> s;

    for (int i = 0; i < 10; ++i) {
        s.push(i);
    }

    for (int i = 0; i < 10; ++i) {
        assert(s.front() == i);
        s.pop();
    }

    return 0;
}
