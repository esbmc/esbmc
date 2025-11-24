#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    // No check for argc before using argv[1]
    int size = atoi(argv[1]);

    // No check if size is reasonable before malloc
    int *arr = malloc(size * sizeof(int));

    // Dereferencing NULL if malloc fails
    arr[0] = 42;

    free(arr);
    return 0;
}

