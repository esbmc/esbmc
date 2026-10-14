/* github_6542_is_fresh_param_extent_fail:
 * The extent names a parameter, so at a call site it has to mean the caller's
 * argument. Rebinding the pointer but not the extent leaves the callee's own
 * symbol in the assertion, and the obligation stops tracking what was asked
 * for. Here the caller promises 4096 bytes of a 16-byte object. */
#include <stdlib.h>

void callee(unsigned char *b, unsigned long n) {
  __ESBMC_requires(__ESBMC_is_fresh(b, n));
  __ESBMC_assigns(b[0]);
  __ESBMC_ensures(b[0] == 1);
  b[0] = 1;
}

int main(void) {
  unsigned char *p = (unsigned char *)malloc(16);
  __ESBMC_assume(p != 0);
  callee(p, 4096);
  return 0;
}
