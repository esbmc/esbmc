#include <stdio.h>

typedef int v4si __attribute__((__vector_size__(16)));
v4si vsi = (v4si){1, 2, 3, 4};

// Should Initialize Correctly
int main() {
   for(int i = 0; i < 4; i++)
      __ESBMC_assert(vsi[i] == i, "The vector should be initialized correctly");
   return 0; 
}