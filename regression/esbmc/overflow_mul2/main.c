#include <limits.h>

int main()
{
  unsigned int ue = UINT_MAX / 2, uf = 3;
  unsigned int mul_wrap = ue * uf;
  return 0;
}
