#include <stdio.h>

typedef int v4si __attribute__((__vector_size__(16)));
v4si vsi = (v4si){1};

// Should Initialize Correctly
int main() {
  for(int i = 0; i < 4; i++)
    __ESBMC_assert(vsi[i] == 1, "The vector should be initialized correctly");
  return 0;
}