/* #9881: a store into an untyped heap object is split into several SSA steps
 * that all carry the call's value; the call still gets one function_return
 * waypoint. */
#include <stdlib.h>
extern unsigned long __VERIFIER_nondet_ulong(void);
struct S
{
  int a;
  unsigned long x;
};
void *alloc(size_t n)
{
  void *p = malloc(n);
  if (!p)
    abort();
  return p;
}
int main(void)
{
  struct S *s = alloc(sizeof(struct S));
  s->x = __VERIFIER_nondet_ulong();
  __ESBMC_assert(s->x != 5, "x is 5");
}
