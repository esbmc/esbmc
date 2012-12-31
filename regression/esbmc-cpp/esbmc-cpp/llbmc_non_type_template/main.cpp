#include <cassert>

template<class T, int N>
class mysequence
{
    T memblock [N];
public:
    void setmember(int x, T value);
    T getmember(int x);
};

template<class T, int N>
void mysequence<T, N>::setmember(int x, T value)
{
    memblock[x] = value;
}

template <class T, int N>
T mysequence<T, N>::getmember(int x)
{
    return memblock[x];
}

int main()
{
    mysequence<int, 5> myints;
    myints.setmember(0, 100);
    myints.setmember(3, 31416);
    assert(myints.getmember(0) == 100);
    assert(myints.getmember(3) == 31416);
    return 0;
}
