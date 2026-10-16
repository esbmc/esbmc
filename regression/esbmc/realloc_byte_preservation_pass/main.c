/* C-suite CORE control for realloc_byte_preservation_knownbug: same shape
 * but no realloc. Verifies that ESBMC's malloc + byte-write + byte-read
 * works correctly when realloc is not in the chain. If this test FAILS,
 * the bug is more pervasive than just realloc.
 */
#include <assert.h>
#include <stdlib.h>

int main(void)
{
  char *p = (char *)malloc(16);
  __ESBMC_assume(p != 0);
  p[0] = 42;
  assert(p[0] == 42);
  return 0;
}
