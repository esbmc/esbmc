/* Negative variant of github_4441 — boundary pin.
 *
 * Same body as github_4441 but without the __assert_fail in reach_error,
 * so control really does reach end-of-main with the heap object
 * unreachable.  The memcleanup check must still fire here, otherwise
 * the fix has over-suppressed the "forgotten memory" claim.
 */

#include <stdlib.h>

int main(void)
{
  char *p = malloc(8);
  if (!p)
    return 0;
  return 0;
}
