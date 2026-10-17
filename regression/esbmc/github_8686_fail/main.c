/* github #8686: strtol stops at the first character that is not a digit of
 * the base, so "19" in base 8 is 1, not 17. */
#include <assert.h>
#include <stdlib.h>
int main(void)
{
  assert(strtol("19", NULL, 8) == 17);
  return 0;
}
