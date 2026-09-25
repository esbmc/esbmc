#include <assert.h>
#include <string.h>
int main()
{
  char src[4] = "abc";
  char dst[4];
  memcpy(dst, src, 4);
  assert(dst[0] == 'a' && dst[1] == 'b' && dst[2] == 'c' && dst[3] == 0);
  return 0;
}
