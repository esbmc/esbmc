#include <ubuntu20.04/kernel_5.15.0-76/include/linux/slab.h>


int main(void)
{
    int * ptr = kmalloc(10 * sizeof(int),1);
    // if(ptr == NULL) return 1;

    // kfree(ptr);

    return 0;
}