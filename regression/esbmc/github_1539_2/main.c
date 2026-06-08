#include <assert.h>

/* The int-to-ptr cast comes BEFORE the string literal is referenced.
 * The pre-registration pass ensures the literal is still visible to
 * the cast so the solver can place it at 0x100, making str0 == str1
 * possible and the assertion unprovable. */
int main()
{
  char *str0 = (char *)0x100;
  char *str1 = "";
  assert(str0 != str1);
}
