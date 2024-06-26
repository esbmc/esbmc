#include <stdio.h>
#include <stdlib.h>

typedef struct {
} Foo;

void Foo_Execute(Foo *self) {
    Foo *ptr = self;
    printf("pointer value: %s\n", *ptr); /* failure here: %s scans for '\0' */
}
int main() {
    Foo *foo = (Foo *)malloc(sizeof(Foo));
    if (foo == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }
    // free(foo);
    Foo_Execute(foo);
    return 0;
}
