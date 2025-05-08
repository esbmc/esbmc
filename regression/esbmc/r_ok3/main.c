#include <assert.h>
#include <stdlib.h>

int main() {
  int N, M;  
  __ESBMC_assume(N > 0 && N < 10);
  __ESBMC_assume(M < N);
  char *arr = malloc(N); 
  assert(__ESBMC_r_ok(arr, M));
}
