// standard h files
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>


extern uint32_t global_var2;  // uninitialized global var.
extern unsigned int global_var3;

int main(void) { 
    uint64_t local_var_addr = (uint32_t)global_var2;  
    global_var3 = *((uint32_t *)local_var_addr);
}
