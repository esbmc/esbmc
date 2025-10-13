#include <string.h>
#include <assert.h>

#define T unsigned long long

int main()
{
  // basic memcpy
  T a0, a1;
  memcpy(&a0, &a1, sizeof(T));
  assert(a0 == a1);

  // memcpy multiple sources
  T b0, b1, b2;
  T b_cond0;

  memcpy(&b0, b_cond0 ? &b1 : &b2, sizeof(T));
  assert(b_cond0 ? b0 == b1 : b0 == b2);

  // memcpy multiple destinations
  T c0, c1, c2;
  T c_cond0;
  memcpy(c_cond0 ? &c0 : &c1, &c2, sizeof(T));
  assert(c_cond0 ? c0 == c2 : c1 == c2);

  // memcpy multiple sources and destinations
  T d0, d1, d2, d3;
  T d_cond0, d_cond1;
  memcpy(d_cond0 ? &d0 : &d1, d_cond1 ? &d2 : &d3, sizeof(T));
  if (d_cond0)
    assert(d_cond1 ? d0 == d2 : d0 == d3);
  else
    assert(d_cond1 ? d1 == d2 : d1 == d3);

  // memcpy with different offsets
  T e0, e1;
  char *e_ptr0 = (char*) &e0;
  char *e_ptr1 = (char*) &e1;
  memcpy(e_ptr0+1, e_ptr1+2, sizeof(char));
  assert(e_ptr0[1] == e_ptr1[2]);
  memcpy(e_ptr0 + 4, e_ptr1 + 3, sizeof(char));
  assert(e_ptr0[4] == e_ptr1[3]);
  return 0;
}
