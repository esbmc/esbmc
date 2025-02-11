
#define PAGE_SIZE 4096 // page size
#define USER_MEMORY_SPACE 10000 //only for simulate the user memory
#define KERNEL_MEMORY_SPACE 10000// only for simulate the kernel memory
extern char user_memory[USER_MEMORY_SPACE]; //mock user memory
extern char kernel_memory[KERNEL_MEMORY_SPACE];//mock user memory
//simulate copy_to_user function in kernel space 
unsigned long copy_to_user(void* to, void* from, unsigned long size);
unsigned long copy_from_user(void* to, void* from, unsigned long size);


    
