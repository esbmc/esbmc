/* SV-COMP'22 c/memsafety/cmp-freed-ptr.c */

#include <stdlib.h>
#include <stdint.h>

int main() {

    struct T {
        struct T* next;
    };

    struct T* x = NULL;
    struct T* y = NULL;

    y = malloc(sizeof(*y));
    intptr_t adressY = (intptr_t) y;

    free(y);

    x = malloc(sizeof(*x));
    intptr_t adressX = (intptr_t) x;

    if (adressX == adressY)
    { // if the second malloc returns the same value as the first, I should get here
        assert(0);
    }

    return 0;
}

// predator-regre/test-0238.c
// - comparing freed pointers
// - contributed by Ondra Lengal
