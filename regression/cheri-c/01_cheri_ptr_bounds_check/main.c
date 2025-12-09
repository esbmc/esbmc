#include <stdio.h>
#include <assert.h>
#include <cheri/cheric.h>

int main() {
    // A regular integer array on the stack
    int buffer[4] = {10, 20, 30, 40};

    // Create a CHERI capability pointer to 'buffer',
    // with bounds set to the size of the entire array (4 * sizeof(int))
    int *__capability cap_ptr = cheri_ptr(buffer, sizeof(buffer));

    // In-bounds access
    printf("Within bounds: %d\n", cap_ptr[2]);
    assert(cap_ptr[2] == 30);

    return 0;
}
