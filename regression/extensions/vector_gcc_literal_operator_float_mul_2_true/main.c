#include <stdio.h>

#define test_type double
#define test_length sizeof(test_type)*4

typedef test_type v4si __attribute__((__vector_size__(test_length)));
v4si vsi = (v4si){0, 1, 2, 3};
v4si vsi2 = (v4si){1.5, 2, 0.5, 3};
// Should Initialize Correctly
int main() {
    vsi *= vsi2;
   for(int i = 0; i < 4; i++)
      __ESBMC_assert(vsi[i] == (i*vsi2[i]), "The vector should be initialized correctly");
   return 0; 
}
