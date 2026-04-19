#include <assert.h>
#include <stdlib.h>

int main() {
  int N, M;  
  char *arr = malloc(N);
  assert(__ESBMC_r_ok(arr, M));
}
