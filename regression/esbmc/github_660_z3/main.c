#include <assert.h>

int main()
{
  double d = nondet_double();
  float f = d;
  assert(f == d);
}
