#include <cassert>

struct dim3
  {
    unsigned int x, y, z;

    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
    {
      x = vx;
      y = vy;
      z = vz;
    }
  };

dim3 blockDim;

int main()
{
  assert(blockDim.x == 1);
  assert(blockDim.y == 1);
  assert(blockDim.z == 0); // should be 1
  return 0;
}
