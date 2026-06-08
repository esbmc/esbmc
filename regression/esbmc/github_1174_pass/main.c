#include <assert.h>
#include <stdio.h>

int main()
{
  /* %02X of 0xAB = "AB" (2 chars), so r must equal 2 */
  int r = printf("%02X", (unsigned)0xAB);
  assert(r == 2);
}
