#include <stdlib.h>

extern void __VERIFIER_nondet_memory(void *, size_t);

/* A range starting at the object but covering only its first member is not
 * the whole object: the next member keeps its value. */
struct pair
{
  int a;
  int b;
};

int main()
{
  struct pair p = {1, 2};
  __VERIFIER_nondet_memory(&p.a, sizeof(p.a));
  __ESBMC_assert(p.b == 2, "sibling kept");
  return 0;
}
