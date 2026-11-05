#include <stdlib.h>

extern void __VERIFIER_nondet_memory(void *, size_t);

struct pair
{
  int a;
  int b;
};

int main()
{
  struct pair p = {1, 2};
  __VERIFIER_nondet_memory(&p.a, sizeof(p.a));
  /* the havocked member can take any value */
  __ESBMC_assert(p.a == 1, "member havocked");
  return 0;
}
