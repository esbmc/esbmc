/* github #8702: strtod("0.3") is the double nearest 0.3, not 3 * 0.1. */
#include <assert.h>
#include <stdlib.h>

int main(void)
{
  assert(strtod("0.3", NULL) == 0.30000000000000004);
  return 0;
}
