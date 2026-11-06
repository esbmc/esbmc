#include <assert.h>
#include <stddef.h>

extern void __VERIFIER_nondet_memory(void *ptr, size_t size);
extern int __VERIFIER_nondet_int(void);

struct pair
{
  int x, y;
};

int main()
{
  /* The SUCCESSFUL twin of testcomp_gen_nondet_memory: the assertion is
   * unreachable, so no test case is written. */
  struct pair p = {0, 0};
  __VERIFIER_nondet_memory(&p, sizeof(p));
  int k = __VERIFIER_nondet_int();
  if (p.x == 0x01020304 && p.y == -1 && p.x == p.y)
    assert(0);
  return 0;
}
