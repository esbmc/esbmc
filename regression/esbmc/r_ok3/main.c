#include <assert.h>
#include <stdlib.h>

int main() {
  int N, M;  
  __ESBMC_assume(N > 0 && N < 10);
  char *arr = malloc(N);
  __ESBMC_assume(M > 0 && M < N);
  assert(__ESBMC_r_ok(arr, M));
}
