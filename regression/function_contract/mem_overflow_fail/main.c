/* mem_overflow_fail:
 * requires does NOT constrain inputs, so a + b may overflow signed int.
 * --overflow-check reports the signed-overflow violation inside the body.
 *
 * Expected: VERIFICATION FAILED (arithmetic overflow)
 */
#include <stddef.h>

int unchecked_add(int a, int b)
{
  __ESBMC_ensures(__ESBMC_return_value == a + b);
  return a + b; /* overflow when a = INT_MAX and b > 0 */
}

int main() { return 0; }
