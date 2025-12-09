#include <ubuntu20.04/kernel_5.15.0-76/include/linux/slab.h>
int main(void)
{
    void * ptr =  kmalloc(24,81);

    return 0;
}