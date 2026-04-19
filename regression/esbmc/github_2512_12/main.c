#include <stddef.h>
#include <stdint.h>
#include <assert.h>

int main(void) {
    struct a {
        int b;
        int c;
        int d;
    } e;

    // Compute pointer to member 'd' safely using offsetof
    char *base = (char *)&e;                  // start from base address of struct
    int *ptr_d = (int *)(base + offsetof(struct a, d));

    // Write safely through pointer
    *ptr_d = 3;

    // Assert the value
    assert(e.d == 3);

    return 0;
}

