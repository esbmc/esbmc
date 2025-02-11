#include <math.h>
#include <assert.h>

unsigned char T(int n)
{
  assert(n >= 1);

  if(n==1)
    return 1;
  else if (n > 1)
    return (1 + T(n/2));
  
  return 0;
}

int main()
{
  // base case
  const unsigned char c = 2;
  unsigned char result1 = T(2);
  unsigned char result2 = log2(2);
  assert(result1 <= c*result2);
  
  // step case
  unsigned char n = nondet_uchar();
  __ESBMC_assume(n>=2);
  unsigned char result3 = T(n);
  unsigned char result4 = log2(n/2)+1;
  assert(result3 <= c*result4);
  
  return 0;
}
