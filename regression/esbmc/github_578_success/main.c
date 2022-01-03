#include <stdint.h>
#include <assert.h>
int main()
{
  uint64_t x = 2;
  x <<= 63;
  x <<= 63;
  assert(x == 0);
  return 0;
}
