/* assume_nonnull_valid_no_null_req_pass:
 * No explicit requires(p != NULL) in the contract: __ESBMC_is_fresh alone
 * must provide a non-null, valid pointer.
 */
#include <stddef.h>

typedef struct { int x; } S;

void f(S *p)
{
  __ESBMC_requires(__ESBMC_is_fresh(p, sizeof(*p)));
  __ESBMC_ensures(p->x == 99);

  p->x = 99;
}

int main()
{
  S s;
  f(&s);
  return 0;
}
