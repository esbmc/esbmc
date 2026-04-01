/* assume_nonnull_valid_req_content_pass:
 * Verifies that requires() properly constrains the nondet content of
 * the malloc'd struct.  The wrapper does:
 *   p = malloc(sizeof(S))       -> p->x is nondet
 *   ASSUME(p->x == 5)           -> constrains nondet to exactly 5
 *   call body: p->x += 10      -> p->x == 15
 *   ASSERT(p->x == 15)          -> must pass
 *
 * If requires did NOT constrain the nondet value, p->x could be any
 * integer after += 10, and the ensures would not hold in general.
 */
#include <stddef.h>

typedef struct { int x; } S;

void f(S *p)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_requires(p->x == 5);
  __ESBMC_ensures(p->x == 15);

  p->x = p->x + 10;
}

int main()
{
  S s;
  s.x = 5;
  f(&s);
  return 0;
}
