#include <assert.h>

int y()
{
    int f = rand();
    while(f)
        return 2;
}

int x()
{
    int f = rand();
    if (f)
    {
        return 1;
    }
    else if(f == 2)
    {
        return 0;
    }
    else
    {

    }
    x();
    return 2;
}

int main()
{
    int a,b;
    int f = a>b;//rand();
    if (f)
    {
        f = x();
    }
    else
    {
        f = x();
    }

    while(a!=b)
    {
        for(int i = 0 ; i<10;i++)
        {

        }
        a= b;
    }

    f = 2;
}
