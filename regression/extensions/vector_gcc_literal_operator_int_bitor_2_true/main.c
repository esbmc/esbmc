#include <stdio.h>

#define test_type int
#define test_length sizeof(test_type)*4

typedef test_type v4si __attribute__((__vector_size__(test_length)));
v4si vsi = (v4si){1, 2, 3, 4};
v4si vsi2 = (v4si){4, 3, 2, 1};

// Should Initialize Correctly
int main() {
    v4si vsi3 = vsi2 | vsi;
   for(int i = 0; i < 4; i++)
      __ESBMC_assert(vsi3[i] == (vsi2[i] | vsi[i]), "The vector should be initialized correctly");
   return 0; 
}
