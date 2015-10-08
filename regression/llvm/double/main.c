#include <assert.h>
//#include <stdio.h>

int main()
{
  long double lld;
  float a[4] = {1.0,2.3f,3, lld};
  float b = 3.90;
  float c = 3.90f;
  double d = 3.5;
  double e = 3.5f;
  
  assert(a[0] == 1.0);
  assert(a[1] == 2.30);
  assert(a[2] == 3);

  return 0;
}
