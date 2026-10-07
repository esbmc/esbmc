#include <stdint.h>

/* As github_7707-assume, with b at an odd offset: an aligned address is not an
 * aligned offset when the packed object's base is free (#7707). */

struct __attribute__((packed)) S
{
  char a;
  uint64_t b;
};

int main(void)
{
  struct S sp;
  uint64_t *p = (uint64_t *)&sp.b;
  __ESBMC_assume(((uintptr_t)p & 7u) == 0);
  uint64_t z = *p;
  (void)z;
  return 0;
}
