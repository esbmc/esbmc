#include <stdlib.h>

int main() {
    int *ptr = malloc(sizeof(int));
    int choice;

    if (choice == 1) {
        free(ptr);
    } else {
        *ptr = 42;        // potential NULL deref
    }

    return 0;
}
