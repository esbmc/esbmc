/* Alias write to the COUNTER's entry value before the loop. */
#include <stdint.h>
#include <assert.h>
int main(void){ uint32_t n; __ESBMC_assume(n>=2&&n<=4);
  uint64_t i = 0; uint64_t *p = &i; *p = 1;
  uint64_t s = 0;
  while (i < n) { s = s + 1; i++; }
  assert(s == n - 1); return 0; }
