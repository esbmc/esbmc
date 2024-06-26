#include <stdbool.h>

int a[20];

bool foo(int il)
{
    return a[1];
}

int main()
{
    int il;
    for (il = 0; foo(il) && il < 10; ++il)
    {
    }
    // if(1 && 2 && 3 && 4);

    for (il = 0; il < 10 && foo(il); ++il)
    {
    }
}