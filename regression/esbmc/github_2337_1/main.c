#include <stdio.h>
#include <stdlib.h>

int maximum(int x, int y) {
    return x ^ ((x ^ y) & -(x < y));
}
int main() {
   
    unsigned int x = 15, y = 20;
    printf("    [*] The maximum of %d and %d is %d\n", 32, 56, maximum(32, 56));
    return 0;
}
