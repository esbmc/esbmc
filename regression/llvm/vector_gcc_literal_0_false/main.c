#include <stdio.h>

// Only sizes that are positive power-of-two multiples of the base type size are currently allowed. 
typedef int v4si __attribute__((__vector_size__(1)));
v4si vsi = (v4si){1, 2, 3, 4};

int main() {
    
   return 0; 
}