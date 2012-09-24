#include <cassert>
#include <algorithm>
#include <deque>

int op_double(int i) { return i + i; }

int main ()
{
    std::deque<int> d;
    std::deque<int> d2;

    for (int i = -10; i <= 10; ++i) {
        d.push_back(i);
    }

    std::transform(d.begin(), d.end(), d2.begin(), op_double);

    for (int i = 0; i <= 20; ++i) {
        assert(d2[i] == (i - 10) << 1);
    }

    return 0;
}
