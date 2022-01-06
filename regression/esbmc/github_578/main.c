#include <stdint.h>
#include <assert.h>
int main()
{
  uint32_t x = 2;
  x <<= 31;
  x <<= 31;
  assert(x == 0);
  return 0;
}
