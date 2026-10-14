#include <assert.h>
#include <string.h>

/* A memset that starts inside one element or member and runs into the next
 * writes only the bytes left in the first. */
struct S
{
  unsigned x;
  unsigned y;
};

int main()
{
  unsigned a[2] = {0, 0};
  memset((char *)a + 3, 0x11, 2);
  assert(a[0] == 0x11000000u && a[1] == 0x00000011u);

  struct S s = {0, 0};
  memset((char *)&s + 3, 0x22, 2);
  assert(s.x == 0x22000000u && s.y == 0x00000022u);
  return 0;
}
