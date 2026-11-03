#include <stdint.h>

/* The counterpart to github_7707-array: b sits at an odd offset, and the
 * program aligns the address it loads from, so the claim on the whole address
 * holds where one on the offset alone would not. */

struct __attribute__((packed)) S
{
  char a;
  uint64_t b;
};

struct S arr[4];

int main(void)
{
  uint64_t *p = (uint64_t *)&arr[1].b;
  __ESBMC_assume(((uintptr_t)p & 7u) == 0);
  uint64_t z = *p;
  (void)z;
  return 0;
}
