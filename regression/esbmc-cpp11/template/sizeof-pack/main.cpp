#include <assert.h>

template <int... a>
struct b
{
    int c() { return sizeof...(a); }
};

int main()
{
    b<0, 1, 2, 3> d;
    int theSize = d.c();
    assert(theSize == 4);
}