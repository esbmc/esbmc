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
  __ESBMC_assert(g.flag != 42, "flag can be havocked to 42");
  return 0;
}
