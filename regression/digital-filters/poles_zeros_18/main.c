#include<assert.h>

int main()
{
  float a[] =  {

   1.000000000000000,  -2.000000000000000,   1.968750000000000,  -2.000000000000000,   1.000000000000000


};
 float b[] =  {
     0,     0 ,    0  ,   0  ,   0
     };
  assert(__ESBMC_check_stability(a, b));
  return 0;
}
