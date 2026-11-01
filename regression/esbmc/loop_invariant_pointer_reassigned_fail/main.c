#include <stdlib.h>
__ESBMC_logic long X(const char *p, long r) { return r <= 0 ? 0 : 1 + X(p + 1, r - 1); }
int main() {
  char *s = malloc(4), *t = malloc(4);
  const char *p = s;
  int i = 0;
  // false after the first iteration: p moves to t
  __ESBMC_loop_invariant(0 <= i && i <= 2 && __ESBMC_same_object(p, s));
  for (; i < 2; i++) p = t;
  return 0;
}
