#include <stddef.h>
extern void __VERIFIER_nondet_memory(void *dst, size_t size);
typedef struct
{
  unsigned long a[512];
  int flag;
} big_t;
big_t g;
int main(void)
{
  __VERIFIER_nondet_memory(&g, sizeof(big_t));
  g.flag = 7;
  __ESBMC_assert(g.flag == 7, "assigned flag survives");

  big_t h;
  h.a[0] = 1;
  __VERIFIER_nondet_memory(&h.a[1], sizeof(unsigned long));
  __ESBMC_assert(h.a[0] == 1, "bytes outside the range are untouched");
  return 0;
}
