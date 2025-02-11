// value-initialize for aggregate
#include <cassert>

struct uint3
{
  unsigned int x, y, z;
};

struct uint3 indexOfThread[1024];

int main()
{
  assert(indexOfThread[0].x == 0);
  assert(indexOfThread[0].y == 0);
  assert(indexOfThread[0].z == 0);
  return 0;
}
