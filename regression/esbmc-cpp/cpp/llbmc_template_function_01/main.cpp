#include <cassert>

template <class T>
T getMax(T a, T b)
{
    T res = (a>b ? a : b);
    return res;
}

int main ()
{
    int i=5, j=6, k;
    long l=10, m=5, n;
    k = getMax<int>(i, j);
    n = getMax<long>(l, m);
    assert(k == 6);
    assert(n == 10);
    return 0;
}
