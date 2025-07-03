#include <stdio.h>
#include <assert.h>
#include <cheri/cheric.h>

int main() {
    // A regular integer array on the stack
    int buffer[4] = {10, 20, 30, 40};

    // Create a CHERI capability pointer to 'buffer',
    // with bounds set to the size of the entire array (4 * sizeof(int))
    int *__capability cap_ptr = cheri_ptr(buffer, sizeof(buffer));

    // out of bounds access
    printf("Within bounds: %d\n", cap_ptr[4]);

    return 0;
}
