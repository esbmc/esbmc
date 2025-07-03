#include <assert.h>
#include <stdlib.h>

int main() {
  int N, M;  
  char *arr = malloc(nondet_int() ? N : N+1);
  __ESBMC_assume(N > 0 && M > 0 && M < N); 
  assert(__ESBMC_r_ok(arr, M));
}
