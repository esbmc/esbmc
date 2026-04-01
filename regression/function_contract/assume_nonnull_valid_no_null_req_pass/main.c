/* assume_nonnull_valid_no_null_req_pass:
 * No explicit requires(p != NULL) in the contract.
 * --assume-nonnull-valid alone must provide a non-null, valid pointer
 * (via malloc + ASSUME(p != NULL) injected by the flag).
 *
 * If the flag's malloc/assume mechanism depends on a requires(p != NULL)
 * being present, this test would fail (either vacuous or dereference error).
 */
#include <stddef.h>

typedef struct { int x; } S;

void f(S *p)
{
  /* Intentionally no requires(p != NULL) — relies solely on the flag */
  __ESBMC_ensures(p->x == 99);

  p->x = 99;
}

int main()
{
  S s;
  f(&s);
  return 0;
}
