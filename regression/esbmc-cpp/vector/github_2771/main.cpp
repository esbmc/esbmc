#include <cassert>

int main()
{
    int arr[5] = {0,0,0,0,0};
    int *p = arr;
    int *end = arr + 5;

    while (p != end)
    {
        assert(*p == 0);
        p++;
    }
}

