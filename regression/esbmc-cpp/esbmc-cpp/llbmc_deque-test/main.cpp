#include <cassert>
#include <deque>

int main()
{
    std::deque<int> d;

    for (int i = 0; i < 10; ++i) {
        if (i % 2) {
            d.push_back(i);
        } else {
            d.push_front(i);
        }
    }
    for (int i = 0; i < 10; ++i) {
        if (i % 2) {
            assert(d.back() == 10 - i);
            d.pop_back();
        } else {
            assert(d.front() == 8 - i);
            d.pop_front();
        }
    }

    return 0;
}
