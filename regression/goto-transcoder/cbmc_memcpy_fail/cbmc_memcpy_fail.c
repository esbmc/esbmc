#include <assert.h>
#include <string.h>
int main()
{
  char src[4] = "abc";
  char dst[4];
  memcpy(dst, src, 4);
  assert(dst[0] == 'z'); // dst[0] is 'a', so this must FAIL
  return 0;
}
