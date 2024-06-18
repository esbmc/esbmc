#include <stdbool.h>

int main()
{
    bool a = true;
    bool b = false;

    if ((a ? b ? 1 : 0 : a == b) && a)
        ;
}