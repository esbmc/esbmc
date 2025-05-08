#include <assert.h>

int main() {
  int N, M;  
  char arr[N];
  __ESBMC_assume(N > 0 && M < N); 
  assert(__ESBMC_r_ok(arr, M));
}
