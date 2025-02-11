#include <assert.h>
struct dim3
{
    unsigned int x, y, z;
};

int main() 
{
  dim3 s1= {1,2,3};

  dim3 s2;

  s2 = s1;

  assert(s2.z == 0); //z should be 3

  return 0;
}
