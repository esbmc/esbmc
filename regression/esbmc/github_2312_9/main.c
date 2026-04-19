#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <limits.h>

// Function to check if a string is a valid positive integer
int is_positive_integer(const char *str) {
    if (str == NULL || *str == '\0') return 0;  // Null or empty string is invalid
    while (*str) {
        if (!isdigit((unsigned char)*str)) return 0;  // Non-numeric character found
        str++;
    }
    return 1;  // Valid positive integer
}

int main(int argc, char *argv[]) {
    if (argc < 2) {  // Ensure an argument is provided
        fprintf(stderr, "Usage: %s <positive integer>\n", argv[0]);
        return 1;
    }

    if (!is_positive_integer(argv[1])) {  // Validate input is a positive integer
        fprintf(stderr, "Error: Argument must be a positive integer.\n");
        return 1;
    }

    // Convert argument to integer safely
    long size = strtol(argv[1], NULL, 10);

    // Prevent negative or unreasonably large allocations
    if (size <= 0 || size > INT_MAX / sizeof(int)) {
        fprintf(stderr, "Error: Invalid size. Must be between 1 and %lu.\n", INT_MAX / sizeof(int));
        return 1;
    }

    // Allocate memory safely
    int *arr = malloc(size * sizeof(int));
    if (!arr) {  // Check if malloc failed
        fprintf(stderr, "Error: Memory allocation failed.\n");
        return 1;
    }

    // Initialize array safely
    arr[0] = 42;
    printf("First element: %d\n", arr[0]);

    // Free allocated memory
    free(arr);

    return 0;
}

