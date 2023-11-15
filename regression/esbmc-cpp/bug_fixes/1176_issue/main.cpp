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

int main() 
{
  assert(dim3(3).x == 3);
  
  assert(dim3(3).y == 1);

  assert(dim3(3).z == 1);

  return 0;
}
