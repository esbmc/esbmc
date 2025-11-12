#include <assert.h>
 
int main()
{
  float a = nondet_float(), b = nondet_float();
  assert(a == b);
  return 0;
}
