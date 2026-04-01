/* assume_nonnull_valid_fail: Proves --assume-nonnull-valid is non-vacuous.
 *
 * The function sets p->x = 9999, but the ensures clause requires p->x == 10.
 * This MUST be caught as VERIFICATION FAILED.
 *
 * Before the fix, this was VERIFICATION SUCCESSFUL (vacuous) because
 * add_pointer_validity_assumptions() emitted ASSUME(valid_object(nil)) = ASSUME(false),
 * killing all paths before the function was ever verified.
 *
 * After the fix (p = malloc(sizeof(S)) instead of ASSUME), the pointer is
 * properly allocated and the ensures violation is detected.
 */

#include <stddef.h>

typedef struct
{
  int x;
} S;

void f(S *p)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_ensures(p->x == 10);
  p->x = 9999; /* WRONG: ensures requires x == 10 */
}

int main()
{
  S s;
  f(&s);
  return 0;
}
