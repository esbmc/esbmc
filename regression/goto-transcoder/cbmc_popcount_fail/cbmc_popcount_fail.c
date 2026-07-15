#include <assert.h>
int main()
{
  unsigned x;
  __CPROVER_assume(x == 0xFFu);
  assert(__builtin_popcount(x) == 7); /* wrong: popcount(0xFF)==8 */
  return 0;
}
