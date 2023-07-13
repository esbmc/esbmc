#include "asm.h"
#include "extable.h"
#include "page.h"
#include "smap.h"
#include "../linux/compiler.h"
#include "../linux/string.h"
#include "../linux/kasan-checks.h"

#define PAGE_SIZE 4096 // page size
#define USER_MEMORY_SIZE 0x100000000 //only for simulate the user memory
#define KERNEL_MEMORY_SIZE 0x100000000 // only for simulate the kernel memory

char user_memory[USER_MEMORY_SIZE]; //mock user memory
char kernel_memory[KERNEL_MEMORY_SIZE];//mock user memory

//simulate copy_to_user function in kernel space 
unsigned long copy_to_user(void* to, void* from, unsigned size);
unsigned long copy_from_user(void* to, void* from, unsigned size);


    
