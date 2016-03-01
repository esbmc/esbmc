#include <assert.h>

int main()
{
  _Bool b = 0;
  int x;
  float y;

  int z1 = b==0 ? x : y;
  float z5 = b==0 ? x : y;

  assert(z1 == z5);

  int z2 = b ? x : y;
  float z6 = b ? x : y;

  assert(z2 == (int)z6);

  int z3 = b==0 ? : y;
  float z7 = b==0 ? : y;

  int z4 = b ? : y;
  float z8 = b ? : y;

  float z9 = x == 2? z5 : z8;
}
