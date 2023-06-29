#include "stdlib.c"
int main(void)
{
    void * ptr =  kmalloc(-100,2);

    return 0;
}