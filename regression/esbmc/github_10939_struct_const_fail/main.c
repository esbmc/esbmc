#include <assert.h>
struct SF { unsigned int u; };
int main(void)
{
  float f = 1.0f;
  struct SF v = *(struct SF *)&f;
  assert(v.u == 0x3f800001u);
  return 0;
}
