#include <ubuntu20.04/kernel_5.15.0-76/include/asm/uaccess.h>


int main()
{
    // Non-deterministic choice of indices
    int kernel_index = nondet_int();
    int user_index = nondet_int();

    // Constrain the indices to be within valid ranges
    __ESBMC_assume(kernel_index >= 0 && kernel_index < KERNEL_MEMORY_SPACE);
    __ESBMC_assume(user_index >= 0 && user_index < USER_MEMORY_SPACE);

    char* kernel_addr = kernel_memory + kernel_index;
    char* user_addr = user_memory + user_index;

    int buffer_size = 4400; 

    copy_from_user(kernel_addr, user_addr, buffer_size);
    return 0;

}