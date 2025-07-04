//FormAI DATASET v1.0 Category: Bitwise operations ; Style: protected
#include <stdio.h>
#include <stdlib.h>

// Function to check if a number is power of 2 or not
int isPowerOfTwo(unsigned int n) {
    return (n && !(n & (n - 1)));
}

// Function to swap two numbers using bitwise XOR
void swap(unsigned int* a_ptr, unsigned int* b_ptr) {
    // If both numbers are not same
    if (*a_ptr != *b_ptr) {
        // Perform XOR operation
        *a_ptr ^= *b_ptr;
        // Assign the result to the first number
        *b_ptr ^= *a_ptr;
        // Assign the result to the second number
        *a_ptr ^= *b_ptr;
    }
}

// Function to count the number of set bits in an integer
int countSetBits(unsigned int n) {
    int count = 0;
    while (n) {
        n &= (n - 1);
        count++;
    }
    return count;
}

// Function to get the maximum of two numbers without using any comparison operator
int maximum(int x, int y) {
    return x ^ ((x ^ y) & -(x < y));
}

int main() {
    unsigned int x = 15, y = 20;

    // Check if x is power of 2 or not
    if (isPowerOfTwo(x))
        printf("%u is power of 2\n", x);
    else
        printf("%u is not power of 2\n", x);

    // Swap the values of x and y
    printf("Before swap: x = %u, y = %u\n", x, y);
    swap(&x, &y);
    printf("After swap: x = %u, y = %u\n", x, y);

    // Count the number of set bits in x
    printf("Number of set bits in %u is %d\n", x, countSetBits(x));

    // Get the maximum of two numbers
    printf("The maximum of %d and %d is %d\n", 32, 56, maximum(32, 56));
    
    return 0;
}


