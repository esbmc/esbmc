#include <assert.h>
union UF { unsigned int u; };
int main(void)
{
  float f = 1.0f;
  union UF v = *(union UF *)&f;
  assert(v.u == 0x3f800000u);
  return 0;
}
