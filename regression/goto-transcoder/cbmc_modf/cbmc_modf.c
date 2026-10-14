#include <assert.h>
#include <math.h>
int main()
{
  double ip;
  double fp = modf(3.75, &ip); // integer 3.0, fraction 0.75 (exact split)
  assert(ip == 3.0 && fp == 0.75);

  float ipf;
  float fpf = modff(-2.25f, &ipf); // sign carried onto both parts
  assert(ipf == -2.0f && fpf == -0.25f);

  long double ipl;
  long double fpl = modfl(5.5L, &ipl); // long double width
  assert(ipl == 5.0L && fpl == 0.5L);
  return 0;
}
