#include <cassert>
#include <algorithm>
#include <deque>

struct MyClass
{
    bool operator()(int i, int j)
    {
        return i<j;
    }
} MyLess;

int main ()
{
    int myints[] = {32,71,12,45};
    std::deque<int> d(myints, myints+4);

    std::sort(d.begin(), d.end(), MyLess);

    for (unsigned int i = 0; i < 3; ++i) {
        assert(d[i] <= d[i+1]);
    }

    return 0;
}
