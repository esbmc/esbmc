#include <stdio.h>

#define test_type double
#define test_length sizeof(test_type)*4

typedef test_type v4si __attribute__((__vector_size__(test_length)));
v4si vsi = (v4si){0, 1, 2, 3};

// Should Initialize Correctly
int main() {
    vsi = 1.5 * vsi;
   for(int i = 0; i < 4; i++)
      __ESBMC_assert(vsi[i] == i*1.5, "The vector should be initialized correctly");
   return 0;
}
