/* A union read over a struct of scalars at constant offset 0 (#10939). */
#include <assert.h>
struct cannot_drop { unsigned int c; };
union chunk { unsigned char bytes[4]; };
int main(void)
{
  struct cannot_drop x = {0x11223344u};
  union chunk a = *(union chunk *)&x;
  assert(a.bytes[3] == 0x11);
  return 0;
}
