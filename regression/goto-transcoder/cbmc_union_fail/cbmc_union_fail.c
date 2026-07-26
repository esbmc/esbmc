#include <assert.h>

union U
{
  int i;
  float f;
};

int main()
{
  union U u;
  u.i = 42;
  assert(u.i == 0);
  return 0;
}
