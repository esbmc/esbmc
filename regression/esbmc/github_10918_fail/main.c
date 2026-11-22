/* #10918: an object may still lie wholly above 2^63, where its addresses are
 * negative as intptr_t. */
#include <assert.h>
#include <stdint.h>
int main(void)
{
  uint64_t g[4];
  assert((intptr_t)&g[0] >= 0);
  return 0;
}
