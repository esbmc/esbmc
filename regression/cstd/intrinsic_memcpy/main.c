#include <string.h>
#include <assert.h>

int main()
{
  // basic memcpy
  unsigned long long a0, a1;
  memcpy(&a0, &a1, sizeof(unsigned long long));
  assert(a0 == a1);

  // memcpy multiple sources
  unsigned long long b0, b1, b2;
  unsigned long long b_cond0;

  memcpy(&b0, b_cond0 ? &b1 : &b2, sizeof(unsigned long long));
  assert(b_cond0 ? b0 == b1 : b0 == b2);

  // memcpy multiple destinations
  unsigned long long c0, c1, c2;
  unsigned long long c_cond0;
  memcpy(c_cond0 ? &c0 : &c1, &c2, sizeof(unsigned long long));
  assert(c_cond0 ? c0 == c2 : c1 == c2);

  // memcpy multiple sources and destinations
  unsigned long long d0, d1, d2, d3;
  unsigned long long d_cond0, d_cond1;
  memcpy(d_cond0 ? &d0 : &d1, d_cond1 ? &d2 : &d3, sizeof(unsigned long long));
  if (d_cond0)
    assert(d_cond1 ? d0 == d2 : d0 == d3);
  else
    assert(d_cond1 ? d1 == d2 : d1 == d3);

  return 0;
}
