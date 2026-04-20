#include <assert.h>

/* str1 registered BEFORE the int-to-ptr cast: the cast result must include
 * str1 as a possible match. The solver can place str1 at 0x100, making
 * str0 == str1 possible, so the assertion cannot be proven. */
int main()
{
  char *str1 = "";
  char *str0 = (char *)0x100;
  assert(str0 != str1);
}
