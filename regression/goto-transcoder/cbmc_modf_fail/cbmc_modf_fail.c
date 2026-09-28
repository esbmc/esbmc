#include <assert.h>
#include <math.h>
int main()
{
  double ip;
  double fp = modf(3.75, &ip); // fraction is 0.75, not 0.5
  assert(fp == 0.5);           // confirms the value is really computed
  return 0;
}
