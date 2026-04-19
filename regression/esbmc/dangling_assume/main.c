#include <stdlib.h>

int main() {
    int *ptr = (int*) malloc(sizeof(int));
    *ptr = 42;
    free(ptr);
    *ptr = 50;

    return 0;
}

