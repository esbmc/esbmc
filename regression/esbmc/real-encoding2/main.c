#include <assert.h>
#include <math.h>

int main()
{
  float x = NAN;
  _Bool y = nondet_bool();

  if (y)
    assert(x != x); // should hold

  assert (y==0 || y==1); 

  return 0;
}

