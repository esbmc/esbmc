#include<assert.h>

int main()
{
  float a[] =  {

     0,     0,     0 ,    0 ,    0


};
 float b[] =  {

     0 ,    0 ,    0  ,   0  ,   0
     };
  assert(__ESBMC_check_stability(a, b));
  return 0;
}
