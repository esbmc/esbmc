#include<assert.h>

int main()
{
  float a[] =  {

   1.000000000000000,  -0.375000000000000,   0.187500000000000


};
 float b[] =  {

   0.406250000000000,  -0.781250000000000,   0.406250000000000
   };
  assert(__ESBMC_check_stability(a, b));
  return 0;
}
