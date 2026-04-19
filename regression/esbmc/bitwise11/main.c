#include <stdio.h>
#include <stdint.h>
#include <string.h>

int main() {
    float f = 3.14f;
    uint32_t bits;

    memcpy(&bits, &f, sizeof(f)); // Copy float's binary representation to an integer

    bits ^= 0x80000000; // Flip the sign bit (example operation)

    memcpy(&f, &bits, sizeof(bits)); // Copy back to float

    printf("Modified float: %f\n", f);
    return 0;
}
