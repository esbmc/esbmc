#include <assert.h>

/* The int-to-ptr cast is made BEFORE str1 is registered.
 * With the fix, pre_solve() defers enumeration so str1 is still visible.
 * The solver can place str1 at 0x100, making str0 == str1 possible,
 * so the assertion cannot be proven. */
int main()
{
  char *str0 = (char *)0x100;
  char *str1 = "";
  assert(str0 != str1);
}
