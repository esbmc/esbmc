#include <stdio.h>

typedef int v4si __attribute__((__vector_size__(16)));


// Should Initialize Correctly
int main() {
  int a = __VERIFIER_nondet_int();
  __VERIFIER_assume(a == 4);
  v4si vsi = (v4si){1, 2, 3, a};
  for(int i = 0; i < 4; i++)
    __ESBMC_assert(vsi[i] == i+1, "The vector should be initialized correctly");
  return 0;
}