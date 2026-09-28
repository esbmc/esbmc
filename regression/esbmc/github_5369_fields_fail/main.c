/* Distinct fields read through one unresolvable pointer must stay free to hold
 * different values: dereference() is handed the same pointer for both and
 * distinguishes them by its lexical offset, so a fix for #5369 keyed on the
 * pointer and the read type alone merges them. */
#include <assert.h>

unsigned long nondet_ulong(void);

struct S
{
  int a;
  int b;
};

int main(void)
{
  struct S *p = (struct S *)nondet_ulong();
  int x = p->a;
  int y = p->b;
  assert(x == y);
  return 0;
}
