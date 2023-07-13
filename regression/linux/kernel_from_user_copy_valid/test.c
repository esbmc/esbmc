#define USER_MEMORY_SPACE 10000
#define KERNEL_MEMORY_SPACE 10000
char user_memory[USER_MEMORY_SPACE]; 
char kernel_memory[KERNEL_MEMORY_SPACE];

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

    unsigned long size = nondet_int();

    __ESBMC_assume(size < 200);
    __ESBMC_assume(kernel_index + size <= KERNEL_MEMORY_SPACE);
    __ESBMC_assume(user_index + size <= USER_MEMORY_SPACE);

    char kernel_buffer[size];
    memcpy(kernel_addr, kernel_buffer, size); 

    copy_from_user(kernel_addr, user_addr, size);

 
     return 0;


}