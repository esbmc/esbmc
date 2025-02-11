#include <assert.h>

void test()
{

    int x = 1;
    for (int i = 0; i < 2; i++)
        assert(x == 1);
}

int test1()
{
    int x = 1;
    for (int i = 0; i < 2; i++)
        assert(x == 1);
}

int main()
{
    test();
    test1();
}